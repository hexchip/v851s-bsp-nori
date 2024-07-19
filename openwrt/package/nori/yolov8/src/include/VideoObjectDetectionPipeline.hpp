#pragma once

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

class VideoObjectDetectionPipeline {
public:

    struct Config {
        std::string modelFilePath = "";
        unsigned int nnRuntimeMemSize = 17 * 1024 * 1024;
        std::vector<std::string> detectionClasses;
        cv::Size inputImgSize = {640, 640};
        float rectConfidenceThreshold = 0.25f;
        float iouThreshold = 0.45f;
    };

    VideoObjectDetectionPipeline(VideoObjectDetectionPipeline::Config& config);

    ~VideoObjectDetectionPipeline();

    void start();

    void stop();

private:
    class Impl;

    std::unique_ptr<Impl> _pImpl;
};
