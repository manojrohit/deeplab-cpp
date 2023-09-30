# include <image.h>

DIR* PrepareImage::OpenDirectory(std::string image_dir){
    DIR *dir;    
    const char* image_dir_c_str = image_dir.c_str();
    std::cout << "Directory Opened" << ((dir = opendir(image_dir_c_str)) != NULL) << "\n";
    return dir;
}

std::vector<std::string> PrepareImage::ListImagesDir(std::string image_dir){
    DIR *dir = OpenDirectory(image_dir);
    struct dirent *ent;
    std::vector<std::string> file_paths;
    while ((ent = readdir(dir)) != NULL) {
        std::string file_name = ent->d_name;
        std::string file_path = image_dir + "/" + file_name;
        file_paths.push_back(file_path);
    }
    return file_paths;
}