#include <string.h>
#include <tensorflow/cc/saved_model/loader.h>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/tsl/platform/status.h"
#include "yaml-cpp/yaml.h"
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <chrono>
#include "tensorflow/cc/ops/math_ops.h"
#include <string.h>

using namespace tensorflow;
using namespace tensorflow::ops;

class ModelFromPB {
    public:
        ModelFromPB(std::string);
        ~ModelFromPB();
        Status getStatus();
        SavedModelBundle& getbundle();
        void printInputOutput();
        int predictInputs(std::string file_path);
        std::pair<std::string, Tensor> input_tensors_key_values;
        std::pair<std::string, Tensor> training_tensors_key_values;
        std::vector<tensorflow::Tensor>* predict(std::vector<std::pair<std::string, Tensor>>, std::string);
        cv::Mat postprocess();
        void colorLogits(cv::Mat& colorMap, const tensorflow::Tensor& segmentation_logits);

    private:
        Status status;
        YAML::Node config;
        // use it to load a saved TensorFlow model and perform operations such as running inferences or restoring variables. 
        SavedModelBundle bundle;
        std::vector<std::pair<std::string, Tensor>> input_tensors;
        std::vector<tensorflow::Tensor>* output_tensor = new std::vector<tensorflow::Tensor>;
        static void printshape(const TensorInfo&);
        std::string input_tensor_name, output_tensor_name;
        // define the Cityscapes color scheme
        const std::vector<cv::Vec3b> cityscapes_colors = {
            cv::Vec3b(128, 64, 128),   // road
            cv::Vec3b(244, 35, 232),   // sidewalk
            cv::Vec3b(70, 70, 70),     // building
            cv::Vec3b(102, 102, 156),  // wall
            cv::Vec3b(190, 153, 153),  // fence
            cv::Vec3b(153, 153, 153),  // pole
            cv::Vec3b(250, 170, 30),   // traffic light
            cv::Vec3b(220, 220, 0),    // traffic sign
            cv::Vec3b(107, 142, 35),   // vegetation
            cv::Vec3b(152, 251, 152),  // terrain
            cv::Vec3b(70, 130, 180),   // sky
            cv::Vec3b(220, 20, 60),    // person
            cv::Vec3b(255, 0, 0),      // rider
            cv::Vec3b(0, 0, 142),      // car
            cv::Vec3b(0, 0, 70),       // truck
            cv::Vec3b(0, 60, 100),     // bus
            cv::Vec3b(0, 80, 100),     // train
            cv::Vec3b(0, 0, 230),      // motorcycle
            cv::Vec3b(119, 11, 32)     // bicycle
        };
};