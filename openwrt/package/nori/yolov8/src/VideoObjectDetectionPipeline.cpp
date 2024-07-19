#include "VideoObjectDetectionPipeline.hpp"

#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <stdexcept>
#include <fcntl.h> // for open, O_RDWR
#include <fstream>
#include <linux/fb.h>
#include <stdint.h>
#include <linux/videodev2.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h> // for close
#include <functional>
#include <sstream>
#include <iomanip>

#include "YoloV8Processor.hpp"
#include "NeuralNetworkRuntime.hpp"
#include "ThreadSafeQueue.hpp"

class VideoObjectDetectionPipeline::Impl {
private:
    YoloV8Processor yoloV8Processor;
    NeuralNetworkRuntime nnRuntime;

    std::atomic<bool> done;
    ThreadSafeQueue<cv::Mat> preprocessQueue;
    ThreadSafeQueue<cv::Mat> inferenceQueue;
    ThreadSafeQueue<std::vector<float>> resultQueue;
    ThreadSafeQueue<std::vector<YoloV8Processor::Detection>> detectionQueue;

    std::thread captureThread;
    std::thread preprocessingThread;
    std::thread inferenceThread;
    std::thread postprocessingThread;

    std::vector<YoloV8Processor::Detection> currentDetections;

    uint32_t frameCount = 0; 

    static uint64_t get_perf_count()
    {
        struct timespec ts;

        clock_gettime(CLOCK_MONOTONIC, &ts);

        return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * 1000000000);
    }

    static void saveVectorToFile(const std::vector<float>& vec, const std::string& filename) {
        std::ofstream outFile(filename);
        if (!outFile) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        outFile << std::fixed << std::setprecision(8); // 设置精度为8位小数

        for (const float& val : vec) {
            outFile << val << "\n"; // 每个值占一行
        }

        outFile.close();
    }

    void captureFrames()
    {
        cv::VideoCapture videoCapture;

        videoCapture.open(0, cv::VideoCaptureAPIs::CAP_V4L2);

        if (!videoCapture.isOpened())
        {
            throw std::runtime_error("Can't initialize camera capture");
        }

        int fd = open("/dev/fb0", O_RDWR);
        if (fd < 0) {
            throw std::runtime_error("Can't open framebuffer device");
        }
        struct fb_var_screeninfo screen_info;
        if (ioctl(fd, FBIOGET_VSCREENINFO, &screen_info)) {
            close(fd);
            throw std::runtime_error("Can't read fixed screen information");
        }
        close(fd);
  
        cv::Size displaySize = {screen_info.xres, screen_info.yres};

        std::cout << "screen_info.xres: " << screen_info.xres << std::endl;
        std::cout << "screen_info.yres: " << screen_info.yres << std::endl;
        std::cout << "screen_info.xres_virtual: " << screen_info.xres_virtual << std::endl;
        std::cout << "screen_info.yres_virtual: " << screen_info.yres_virtual << std::endl;
        std::cout << "screen_info.xoffset: " << screen_info.xoffset << std::endl;
        std::cout << "screen_info.yoffset: " << screen_info.yoffset << std::endl;
        std::cout << "screen_info.bits_per_pixel = " << screen_info.bits_per_pixel << std::endl;

        cv::Mat frame;

        videoCapture >> frame;

        if (frame.empty())
        {
            throw std::invalid_argument("Error: Can't grab camera frame!");
        }

        if (frame.depth() != CV_8U || frame.channels() != 3) {
            throw std::invalid_argument("Unsupported image format: Must be 8 bits per pixel and 3 channels!");
        }

        preprocessQueue.push(frame.clone());

        cv::resize(frame, frame, displaySize);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2BGR565);

        std::ofstream ofs("/dev/fb0"); // 打开帧缓冲区

        ofs.write(reinterpret_cast<char*>(frame.data), frame.total() * frame.elemSize());

        while (!done.load()) {
            videoCapture >> frame;
            if (frame.empty())
            {
                throw std::invalid_argument("Error: Can't grab camera frame!");
            }

            if (frame.depth() != CV_8U || frame.channels() != 3) {
                throw std::invalid_argument("Unsupported image format: Must be 8 bits per pixel and 3 channels!");
            }

            if(!detectionQueue.isEmpty()) {
                if(preprocessQueue.isEmpty()) {
                    frameCount++;
                    preprocessQueue.push(frame.clone());
                }
                currentDetections = detectionQueue.pop();
            }

            yoloV8Processor.drawBoundingBox(frame, currentDetections);

            if (frame.rows != displaySize.width && frame.cols != displaySize.height) {
                cv::resize(frame, frame, displaySize);
            }

            cv::cvtColor(frame, frame, cv::COLOR_BGR2BGR565);

            ofs.seekp(0);
            ofs.write(reinterpret_cast<char*>(frame.data), frame.total() * frame.elemSize());
        }
    }

    void preprocessFrames()
    {
        while (!done.load()) {
            //BGR
            auto frame = preprocessQueue.pop();

            yoloV8Processor.preProcess(frame);

            // auto tmpFrame = frame.clone();

            inferenceQueue.push(frame);

            // cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2RGB);
            // std::ostringstream filename;
            // filename << "frames/frame_" << std::setw(6) << std::setfill('0') << frameCount << ".jpg";
            // cv::imwrite(filename.str(), tmpFrame);

        }
    }

    void onLoadingInputData(int bufferIndex, void *buffer, NeuralNetworkRuntime::InputDataFormat elementDataFormat) {
   

        if (bufferIndex != 0) {
            throw std::invalid_argument("Invalid buffer index! must to be 0!");
        }

        // BRG
        auto frame = inferenceQueue.pop();

        size_t total = frame.total();

        uint8_t* uint8Buffer = static_cast<uint8_t*>(buffer);

        // BRG to RRR...GGG...BBB...
        int channel = 1;
        for (int i = 0; i < total; i++) {
            uint8Buffer[i + channel * total] = frame.data[i * 3 + channel];
        }

        channel = 2;
        for (int i = 0; i < total; i++) {
            uint8Buffer[i + channel * total] = frame.data[i * 3 + channel];
        }

        channel = 0;
        for (int i = 0; i < total; i++) {
            uint8Buffer[i + channel * total] = frame.data[i * 3 + channel];
        }
    }

    void performInference()
    {
        while (!done.load()) {

            auto results = nnRuntime.run([this](int bufferIndex, void *buffer, NeuralNetworkRuntime::InputDataFormat elementDataFormat){
                this->onLoadingInputData(bufferIndex, buffer, elementDataFormat);
            });

            resultQueue.push(results[0]);
        }
    }

    void processResults()
    {
        while (!done.load()) {
            auto result = resultQueue.pop();

            auto detections = yoloV8Processor.postProcess(CV_32FC1, (void *)result.data());

            detectionQueue.push(detections);
        }
    }

    static YoloV8Processor createYoloV8Processor(VideoObjectDetectionPipeline::Config& config) {
        YoloV8Processor::Config yoloV8ProcessorConfig = {
            .classes = std::move(config.detectionClasses),
            .imgSize = config.inputImgSize,
            .rectConfidenceThreshold = config.rectConfidenceThreshold,
            .iouThreshold = config.iouThreshold
        };
        return YoloV8Processor(yoloV8ProcessorConfig);
    }

    static NeuralNetworkRuntime createNeuralNetworkRuntime(VideoObjectDetectionPipeline::Config& config) {
        NeuralNetworkRuntime::Config nnRuntimeConfig = {
            .isAutoInit = false,
            .modelFilePath = config.modelFilePath,
            .memSize = config.nnRuntimeMemSize
        };

        return NeuralNetworkRuntime(nnRuntimeConfig);
    }

public:
    Impl(VideoObjectDetectionPipeline::Config& config) :
        yoloV8Processor(createYoloV8Processor(config)),
        nnRuntime(createNeuralNetworkRuntime(config)),
        done(false),
        preprocessQueue(1),
        inferenceQueue(1),
        resultQueue(1),
        detectionQueue(1)
    {

    }

    void start() {
        nnRuntime.create();

        captureThread = std::thread([this]() {
            try
            {
                this->captureFrames();
            }
            catch(const std::exception& e)
            {
                std::cerr << "captureThread: " << e.what() << '\n';
            }
        });
        preprocessingThread = std::thread([this]() {
            try
            {
                this->preprocessFrames();
            }
            catch(const std::exception& e)
            {
                std::cerr << "preprocessingThread: " << e.what() << '\n';
            }
        });
        inferenceThread = std::thread([this]() {
            try
            {
                this->performInference();
            }
            catch(const std::exception& e)
            {
                std::cerr << "inferenceThread: " << e.what() << '\n';
            }
        });
        postprocessingThread = std::thread([this]() {
            try
            {
                this->processResults();
            }
            catch(const std::exception& e)
            {
                std::cerr << "postprocessingThread: " << e.what() << '\n';
            }
        });

        captureThread.join();
        preprocessingThread.join();
        inferenceThread.join();
        postprocessingThread.join();
    }

    void stop() {
        std::cerr << "call stop!!!" << std::endl;
        done.store(true);
        nnRuntime.destroy();
    }

    ~Impl()
    {
        if (!done.load()) {
            stop();
        }
        std::cout << "VideoObjectDetectionPipeline destroyed!" << std::endl;
    }
};

VideoObjectDetectionPipeline::VideoObjectDetectionPipeline(VideoObjectDetectionPipeline::Config& config)
    : _pImpl(new Impl(config)) 
{

}

VideoObjectDetectionPipeline::~VideoObjectDetectionPipeline() = default;

void VideoObjectDetectionPipeline::start() {
    _pImpl->start();
} 

void VideoObjectDetectionPipeline::stop() {
    _pImpl->stop();
} 