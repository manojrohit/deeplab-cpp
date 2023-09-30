#include "inference.h"
#include "string.h"

ModelFromPB::ModelFromPB(std::string config_file){

    // Load the YAML file into a node
    config = YAML::LoadFile(config_file);

    // Access the data in the YAML file
    std::string model_dir = config["model_dir"].as<std::string>();

    //  top-level scope in the graph and can be used to build the entire computational graph
    Scope root = Scope::NewRootScope();

    // SessionOptions to specify the target device (e.g., CPU or GPU), the number of threads to use for parallel computation, the logging verbosity level, and other options. 
    SessionOptions session_options;
    session_options.config.mutable_gpu_options()->set_visible_device_list("0");
    tensorflow::ClientSession session(root, session_options);

    // RunOptions structure provides a convenient way to control various aspects of the session's behavior, and is passed as an argument to the Session::Run() method.
    RunOptions run_options;

    // Tag to identify which model is being used. Doesnt change node names or varibale names
    const string saved_model_tag = "serve";
    status = LoadSavedModel(session_options, run_options, model_dir, {saved_model_tag},
                 &bundle);
    std::cout << "Model Load Complete" << "\n";

}

ModelFromPB::~ModelFromPB(){
    delete output_tensor;
}

Status ModelFromPB::getStatus(){
    return status;
}

SavedModelBundle& ModelFromPB::getbundle(){
    return bundle;
}

void ModelFromPB::printshape(const TensorInfo& tensor_info){
    const TensorShapeProto& tensor_shape = tensor_info.tensor_shape();
    std::cout << "\n";
    for (int i = 0; i < tensor_shape.dim_size(); i++) {
        const auto& dim = tensor_shape.dim(i);
        int64 size = dim.size();
        std::cout << size << "  ";
    }
    std::cout << "\n";
    
}
void ModelFromPB::printInputOutput(){
    // "serving_default" is a default signature that is used for serving the model.
    // signature is automatically generated when you export a model using the tf.saved_model.save function in TensorFlow. 
    const tensorflow::SignatureDef& signature_def = bundle.meta_graph_def.signature_def().at("serving_default");
    const auto& inputs = signature_def.inputs();
    const auto& outputs = signature_def.outputs();
    
    std::cout << "----- Input and Output tensors of the Model -----" << std::endl;
    // print input and outputs
    for (const auto& input : inputs) {
        auto tensor_name = input.second.name();
        std::cout << "Input tensor " << tensor_name << std::endl;
        std::cout << "Shape";
        printshape(input.second);
        input_tensor_name = tensor_name;
    }
    

    for (const auto& output : outputs) {
        auto tensor_name = output.second.name();
        std::cout << "Output tensor " << tensor_name << std::endl;
        std::cout << "Shape";
        printshape(output.second);
        output_tensor_name = tensor_name;
    }
    
}

int ModelFromPB::predictInputs(std::string file_path){

        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << file_path << std::endl; 
            return 0;
        }

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // std::cout << "Processing image: " << file_path << std::endl;
        Tensor tensor(DT_FLOAT, TensorShape({1, 1024, 2048, 3}));
        
        // Copy the image data into the tensor
        auto tensor_data = tensor.tensor<float, 4>();
        auto cv_data = image.data;
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                for (int c = 0; c < image.channels(); ++c) {
                    tensor_data(0, y, x, c) = cv_data[image.cols * image.channels() * y  + image.channels() * x + c];
            }
        }
        }
        input_tensors.clear();
        std::string input_key(input_tensor_name);
        input_tensors_key_values = {input_key, tensor};
        input_tensors.emplace_back(input_tensors_key_values);
        std::string old_substr = "/images/";
        std::string new_substr = "/predictions/";
        size_t pos = file_path.find(old_substr);
        if (pos != std::string::npos) {
            file_path.replace(pos, old_substr.length(), new_substr);
        }
        output_tensor = predict(input_tensors, file_path);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
        std::cout << "Predict Function takes : " << elapsed_time.count() << " ms" << std::endl;
        return 1;
}

std::vector<tensorflow::Tensor>* ModelFromPB::predict(std::vector<std::pair<std::string, Tensor>> input_tensors_1, std::string predict_path){
    // Identify the output signature
    // std::vector<std::string> output_tensor_names = {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:3"};
    std::vector<std::string> output_tensor_names = {"StatefulPartitionedCall:0"};
    std::vector<std::string> target_tensor_names = {};

    // Rin the session by feeding input tensornames, tensorvalues, output tensornames, target tensors are not there, so empty list
    auto start_time = std::chrono::high_resolution_clock::now();
    TF_CHECK_OK(bundle.session->Run(input_tensors_1, output_tensor_names, target_tensor_names, output_tensor));
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
    std::cout << "Elapsed time for predict: " << elapsed_time.count() << " ms" << std::endl;
    cv::Mat colormap = postprocess();
    cv::imwrite(predict_path, colormap);
    return output_tensor;
}

void ModelFromPB::colorLogits(cv::Mat& colorMap, const tensorflow::Tensor& segmentation_logits){
    auto segmentation_logits_tensor = segmentation_logits.tensor<float, 4>();
    
    for(int i = 0; i < 1024; i++){
        for(int j = 0; j < 2048; j++){
            std::vector<float> numbers = {};
            for(int k = 0; k < 19; k++){
                numbers.emplace_back(segmentation_logits_tensor(0, i, j, k));
            }
            auto maxIter = std::max_element(numbers.begin(), numbers.end());
            int maxIndex = std::distance(numbers.begin(), maxIter);
            colorMap.at<cv::Vec3b>(i, j) = cityscapes_colors[maxIndex];
        }
    }
    // cv::resize(colorMap, colorMap, cv::Size(512, 10));
}

cv::Mat ModelFromPB::postprocess(){
    
    // std::cout << "Output tensor elements" << output_tensor->size() << "\n";
    // std::pair<std::string, Tensor> output_tensor_1 = output_tensor[0];
    cv::Mat colorMap(1024, 2048, CV_8UC3, cv::Scalar(0, 0, 0));

    for (const auto& tensor : *output_tensor) {
        auto tensor_shape = tensor.shape();
        int dimension_size = tensor_shape.dim_size(3);
        // do something with the size of each dimension
        if (dimension_size == 19){
            auto& segmentation_logits = tensor;
            colorLogits(colorMap, segmentation_logits);
        }
    }
    return colorMap;
}

