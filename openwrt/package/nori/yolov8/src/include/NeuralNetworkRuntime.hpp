#pragma once

#include <vector>
#include <memory>
#include <functional>

class NeuralNetworkRuntime
{
public:
    enum InputDataFormat
    {
        FORMAT_UNKNOWN,
        FORMAT_FP32,
        FORMAT_FP16,
        FORMAT_UINT8,
        FORMAT_INT8,
        FORMAT_UINT16,
        FORMAT_INT16
    };

    typedef std::function<void(int bufferIndex, void *buffer, InputDataFormat elementDataFormat)> LoadingInputDataCallback;

    struct Config
    {
        bool isAutoInit = true;
        std::string modelFilePath = "";
        unsigned int memSize = 17 * 1024 * 1024;
    };

    NeuralNetworkRuntime(Config &config);

    NeuralNetworkRuntime(const NeuralNetworkRuntime &nnRuntime) = delete;
    NeuralNetworkRuntime &operator=(const NeuralNetworkRuntime &other) = delete;

    NeuralNetworkRuntime(NeuralNetworkRuntime &&nnRuntime) noexcept;
    NeuralNetworkRuntime &operator=(NeuralNetworkRuntime &&other) noexcept;

    ~NeuralNetworkRuntime();

    void create();

    std::vector<std::vector<float>> run(const LoadingInputDataCallback& onLoadingInputData);

    void destroy();

private:
    class Impl;

    std::unique_ptr<Impl> _pImpl;
};