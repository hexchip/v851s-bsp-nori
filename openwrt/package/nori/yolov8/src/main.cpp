#include <iostream>
#include <memory>
#include <string>
#include <csignal>
#include <stdexcept>
#include <fstream>
// #include <iomanip>

#include <fcntl.h>
// #include <stdint.h>

// #include <time.h>

// #include <sys/mman.h>
// #include <sys/types.h>
// #include <sys/stat.h>
#include <sys/ioctl.h>

#include <unistd.h>
#include <errno.h>
#include <linux/fb.h>

// #include <sstream>

#include "YoloV8Processor.hpp"
#include "NeuralNetworkRuntime.hpp"

static const char *usage =
    "Usage:\nmodelFilePath classesFilePath [nnRuntimeMemSzie]";

static bool isDone = false;

static void terminate(int sig_no)
{
    std::cout << "Got signal: " << sig_no << ", exiting ..." << std::endl;
	isDone = true;
}

static void install_sig_handler(void)
{
    // signal(SIGBUS, terminate); // 当程序访问一个不合法的内存地址时发送的信号
    // signal(SIGFPE, terminate); // 浮点异常信号
    // signal(SIGHUP, terminate); // 终端断开连接信号
    // signal(SIGILL, terminate); // 非法指令信号
    std::signal(SIGINT, terminate); // 中断进程信号
    // signal(SIGIOT, terminate); // IOT 陷阱信号
    // signal(SIGPIPE, terminate); // 管道破裂信号
    // signal(SIGQUIT, terminate); // 停止进程信号
    // signal(SIGSEGV, terminate); // 无效的内存引用信号
    // signal(SIGSYS, terminate); // 非法系统调用信号
    // signal(SIGTERM, terminate); // 终止进程信号
    // signal(SIGTRAP, terminate); // 跟踪/断点陷阱信号
    // signal(SIGUSR1, terminate); // 用户定义信号1
    // signal(SIGUSR2, terminate); // 用户定义信号2
}

static std::string trim(const std::string& str) {
    auto start = std::find_if_not(str.begin(), str.end(), ::isspace);
    auto end = std::find_if_not(str.rbegin(), str.rend(), ::isspace).base();
    return (start < end ? std::string(start, end) : std::string());
}

int main(int argc, char *argv[])
{

    install_sig_handler();

    if (argc < 3)
    {
        std::cerr << "Arguments count " << argc << " is incorrect!" << std::endl;
        std::cout << usage << std::endl;
        return -1;
    }

    std::string classesFilePath(argv[2]);
    std::ifstream classesFile(classesFilePath);
    std::vector<std::string> classes;
    if (classesFile.is_open()) {
        std::string line;
        while (std::getline(classesFile, line)) {
            classes.push_back(trim(line));
        }

        classesFile.close();
    } else {
        std::cerr << "无法打开文件: " << classesFilePath << std::endl;
        return -1;
    }

    try
    {
        NeuralNetworkRuntime::Config nnRuntimeConfig = {
            .modelFilePath = (const char *)argv[1],
        };

        if (argc > 3) {
            nnRuntimeConfig.memSize = static_cast<unsigned int>(std::stoul(argv[3]));
        }

        auto nnRuntime = NeuralNetworkRuntime(nnRuntimeConfig);

        YoloV8Processor::Config yoloV8ProcessorConfig = {
            .classes = std::move(classes),
            .imgSize = {320, 320},
        };
        auto yoloV8Processor = YoloV8Processor(yoloV8ProcessorConfig);

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

        std::ofstream ofs("/dev/fb0"); // 打开帧缓冲区

        // BRG
        cv::Mat frame;

        while (!isDone)
        {
            videoCapture >> frame;

            auto results = nnRuntime.run(
                [&frame, &yoloV8Processor]
                (int bufferIndex, void *buffer, NeuralNetworkRuntime::InputDataFormat elementDataFormat)
                {
                    if (bufferIndex != 0) {
                        throw std::invalid_argument("Invalid buffer index! must to be 0!");
                    }

                    auto inputFrame = frame.clone(); 

                    yoloV8Processor.preProcess(inputFrame);

                    size_t total = inputFrame.total();

                    int8_t* int8Buffer = static_cast<int8_t*>(buffer);

                    // BGR to RRR...GGG...BBB...
                    // uint8 to int8
                    for(int channel = 2; channel >=0; channel--) {
                        for (int i = 0; i < total; i++) {
                            int8Buffer[i + channel * total] = inputFrame.data[i * 3 + channel] - 128;
                        }
                    }
                }
            );

            const std::vector<float> &result = results[0];

            auto detections = yoloV8Processor.postProcess(CV_32FC1, (void *)result.data());

            yoloV8Processor.drawBoundingBox(frame, detections);

            cv::resize(frame, frame, displaySize);

            cv::cvtColor(frame, frame, cv::COLOR_BGR2BGR565);

            ofs.seekp(0);
            ofs.write(reinterpret_cast<char*>(frame.data), frame.total() * frame.elemSize());
        }
        
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';        
    }
	
    return 0;
}