#include "opencv2/opencv.hpp"

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: MNISTClassifier_onnx_opencv <onnx_model_path> <image_path>" << std::endl;
        return 1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(argv[1]);
    if (net.empty())
    {
        std::cout << "Error: Failed to load ONNX file." << std::endl;
        return 1;
    }
    std::filesystem::path srcPath(argv[2]);

    for (auto& imgPath : std::filesystem::recursive_directory_iterator(srcPath))
    {
        if(!std::filesystem::is_regular_file(imgPath))
            continue;

        const cv::Mat image = cv::imread(imgPath.path().string(), cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            std::cerr << "Error: Failed to read image file." << std::endl;
            continue;
        }

        const cv::Size size(28, 28);
        cv::Mat resized_image;
        cv::resize(image, resized_image, size);

        cv::Mat float_image;
        resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

        cv::Mat input_blob = cv::dnn::blobFromImage(float_image);
        
        net.setInput(input_blob);
        cv::Mat output = net.forward();

        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);
        const int class_id = classIdPoint.x;

        std::cout << "Class ID: " << class_id << std::endl;
        std::cout << "Confidence: " << confidence << std::endl;
        cv::Mat bigImg;
        cv::resize(image,bigImg,cv::Size(128,128));
        auto parentPath = imgPath.path().parent_path();
        auto label = parentPath.filename().string()+std::string("<->")+std::to_string(class_id);
        cv::putText(bigImg, label, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::imshow("img",bigImg);
        cv::waitKey();
    }

    return 0;
}
