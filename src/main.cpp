#include <iostream>
#include <inference.h>
#include <image.h>
#include "yaml-cpp/yaml.h"


int main(int argc, char *argv[]){

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " config.yaml" << std::endl;
        return 1;
    }

    // Read YAML file
    std::string config_file = argv[1];
    YAML::Node config = YAML::LoadFile(config_file);
    std::string image_dir = config["image_dir"].as<std::string>();

    // Get the list of images in the image_dir
    PrepareImage prepare_images;
    std::vector<std::string> file_paths = prepare_images.ListImagesDir(image_dir);

    // Create an instance ModelFromPB class
    ModelFromPB model_from_pb(config_file);
    SavedModelBundle& bundle = model_from_pb.getbundle();

    // Print Input and Outputs of the Neural Network
    model_from_pb.printInputOutput();

    for (std::string& file_path : file_paths) {
        if(!file_path.empty())
            model_from_pb.predictInputs(file_path);
    }

    return 0;
}