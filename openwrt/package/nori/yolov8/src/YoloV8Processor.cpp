#include "YoloV8Processor.hpp"

#include <cmath>
#include <iostream>

#define min(a, b) (((a) < (b)) ? (a) : (b))

class YoloV8Processor::Impl
{
private:
    YoloV8Processor::Config config;
    std::vector<cv::Scalar> colors;
    int strideNum = 0;
    int signalResultNum = 0;

    float scaleRatio = 1;
    int heightPadding = 0;
    int widthPadding = 0;


public:
    Impl(YoloV8Processor::Config &config)
        : config(config), colors(config.classes.size())
    {
        for(int i = 0; i < config.classes.size(); i++) {
            cv::RNG rng(cv::getTickCount());
            colors[i] = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        }

        for (int i = 0; i < 3; i++) {
            int stride = (1 << i) * 8;
            int feat_width = config.imgSize.width / stride;
            int feat_height = config.imgSize.height / stride;
            strideNum += feat_width * feat_height;
        }

        signalResultNum = config.classes.size() + 4;
    }

    Impl(const Impl &other) = delete;
    Impl &operator=(const Impl &other) = delete;

    Impl(Impl &&other) noexcept
        : config(std::move(other.config)),
          colors(std::move(other.colors)),
          strideNum(other.strideNum),
          signalResultNum(other.signalResultNum),
          scaleRatio(other.scaleRatio),
          heightPadding(other.heightPadding),
          widthPadding(other.heightPadding)
    {

    }

    Impl &operator=(Impl &&other) noexcept
    {
        if (this != &other)
        {
            config = std::move(other.config);
            colors = std::move(other.colors);
            strideNum = other.strideNum;
            signalResultNum = other.signalResultNum;
            scaleRatio = other.scaleRatio;
            heightPadding = other.heightPadding;
            widthPadding = other.widthPadding;
        }
        return *this;
    }

    void preProcess(cv::Mat &img)
    {
        int imgHeight = img.rows;
        int imgWidth = img.cols;

        if (imgHeight == config.imgSize.height && imgWidth == config.imgSize.width) {
            return;
        }

        scaleRatio = min( (float)config.imgSize.height / imgHeight, (float)config.imgSize.width / imgWidth);
        scaleRatio = min(scaleRatio, 1.0f);

        int unpaddedImgHeight = static_cast<int>(round(imgHeight * scaleRatio));
        int unpaddedImgWidth = static_cast<int>(round(imgWidth * scaleRatio));

        if (scaleRatio <= 1.0f)
        {   
            cv::Size unpaddedImgSize(unpaddedImgWidth, unpaddedImgHeight);
            cv::resize(img, img, unpaddedImgSize);
        }

        heightPadding = config.imgSize.height - unpaddedImgHeight;
        widthPadding = config.imgSize.width - unpaddedImgWidth;

        float deltaHeight = heightPadding * 0.5f;
        float deltaWidth = widthPadding * 0.5f;
        int top = static_cast<int>(round(deltaHeight - 0.1));
        int bottom = static_cast<int>(round(deltaHeight + 0.1));
        int left = static_cast<int>(round(deltaWidth - 0.1));
        int right = static_cast<int>(round(deltaWidth + 0.1));
        cv::Scalar value(114, 114, 114);
        cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, value);
    }

    std::vector<Detection> postProcess(int dataElementType, void *data)
    {

        cv::Mat mat = cv::Mat(signalResultNum, strideNum, dataElementType, data);
        cv::transpose(mat, mat);

        if (dataElementType != CV_32F)
        {
            mat.convertTo(mat, CV_32F);
        }

        float *rawData = (float *)mat.data;

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect2d> boxes;

        for (int i = 0; i < strideNum; i++)
        {
            float *classesScores = rawData + 4;
            cv::Mat scores(1, config.classes.size(), CV_32FC1, classesScores);
            cv::Point classId;
            double maxClassScore;
            cv::minMaxLoc(scores, NULL, &maxClassScore, NULL, &classId);
            if (maxClassScore > config.rectConfidenceThreshold)
            {
                float cx = rawData[0];
                float cy = rawData[1];
                float width = rawData[2];
                float height = rawData[3];
                float x = cx - 0.5f * width;
                float y = cy - 0.5f * height;

                cv::Rect2d box(x, y, width, height);
                
                boxes.push_back(box);
                confidences.push_back(maxClassScore);
                classIds.push_back(classId.x);
            }
            rawData += signalResultNum;
        }

        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, config.rectConfidenceThreshold, config.iouThreshold, nmsResult, 0.5f);

        std::vector<Detection> detections(nmsResult.size());

        for (int i = 0; i < nmsResult.size(); i++)
        {
            int idx = nmsResult[i];

            Detection detection = {
                .classId = classIds[idx],
                .confidence = confidences[idx],
                .box = boxes[idx]
            };

            detections[i] = detection;
        }

        return detections;
    }

    void drawBoundingBox(cv::Mat &img, std::vector<Detection> &detections)
    {
        float deltaWidth = widthPadding * 0.5f;
        float deltaHeight = heightPadding * 0.5f;

        for (int i = 0; i < detections.size(); i++)
        {

            auto &detection = detections[i];

            std::string label = config.classes[detection.classId] + " " +
                                std::to_string(detection.confidence).substr(0, std::to_string(detection.confidence).size() - 4);

            auto &box = detection.box;

            int x1 = std::round(box.x - deltaWidth) / scaleRatio;
            int y1 = std::round(box.y - deltaHeight) / scaleRatio;
            int x2 = std::round((box.x + box.width - deltaWidth) / scaleRatio);
            int y2 = std::round((box.y + box.height - deltaHeight) / scaleRatio);

            cv::rectangle(
                img,
                cv::Point(x1, y1),
                cv::Point(x2, y2),
                colors[detection.classId],
                6);

            cv::putText(
                img,
                label,
                cv::Point(x1 - 10, y1 - 10),
                cv::FONT_HERSHEY_PLAIN,
                3,
                colors[detection.classId],
                4,
                cv::LineTypes::LINE_AA);
        }
    }
};

YoloV8Processor::YoloV8Processor(Config &config)
    : _pImpl(new Impl(config))
{
}

YoloV8Processor::YoloV8Processor(YoloV8Processor &&other) noexcept
    : _pImpl(std::move(other._pImpl))
{
    other._pImpl = nullptr;
}

YoloV8Processor &YoloV8Processor::operator=(YoloV8Processor &&other) noexcept
{
    if (this != &other)
    {
        _pImpl = std::move(other._pImpl);
        other._pImpl = nullptr;
    }
    return *this;
}

void YoloV8Processor::preProcess(cv::Mat &img)
{
    _pImpl->preProcess(img);
}

std::vector<YoloV8Processor::Detection> YoloV8Processor::postProcess(int dataElementType, void *data)
{
    return _pImpl->postProcess(dataElementType, data);
}

void YoloV8Processor::drawBoundingBox(cv::Mat &img, std::vector<Detection> &detections)
{
    _pImpl->drawBoundingBox(img, detections);
}

YoloV8Processor::~YoloV8Processor() = default;