#pragma once

#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>

class YoloV8Processor
{
public:
    struct Config
    {
        std::vector<std::string> classes;
        cv::Size imgSize = {.width = 640, .height = 640};
        float rectConfidenceThreshold = 0.25f;
        float iouThreshold = 0.45f;
    };

    struct Detection
    {
        int classId;
        float confidence;
        cv::Rect2d box;
    };

    YoloV8Processor(Config &config);

    YoloV8Processor(const YoloV8Processor &yoloV8Processor) = delete;
    YoloV8Processor &operator=(const YoloV8Processor &other) = delete;

    YoloV8Processor(YoloV8Processor &&yoloV8Processor) noexcept;
    YoloV8Processor &operator=(YoloV8Processor &&other) noexcept;

    void preProcess(cv::Mat &img);

    std::vector<Detection> postProcess(int dataElementType, void *data);

    void drawBoundingBox(cv::Mat &img, std::vector<Detection> &detections);

    ~YoloV8Processor();

private:
    class Impl;

    std::unique_ptr<Impl> _pImpl;
};