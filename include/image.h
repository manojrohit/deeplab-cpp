#include <opencv2/opencv.hpp>
#include <dirent.h>

class PrepareImage {

    public:
        void GetNewImage(std::string filename);
        std::vector<std::string> ListImagesDir(std::string image_dir);
    
    private:
        void ConvertRGB();
        cv::Mat GetImage();
        void ConvertMatFloatTensor();
        cv::Mat image;
        std::string image_dir;
        DIR* OpenDirectory(std::string image_dir);
        
};