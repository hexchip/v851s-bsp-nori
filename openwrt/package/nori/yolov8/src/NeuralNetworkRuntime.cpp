#include "NeuralNetworkRuntime.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <exception>
#include <algorithm>
#include <utility>
#include <cstring>

#include <stdint.h>

#include <time.h>

#include <stdio.h>
#include <memory.h>
#include <stdlib.h>

#include "vip_lite.h"

#include "NullPointerException.hpp"
#include "VipStatusException.hpp"

#include <sstream>
#include <iomanip>


class NeuralNetworkRuntime::Impl
{
private:
    enum BufferType
    {
        TYPE_UNDEFINED,
        TYPE_IN,
        TYPE_OUT
    };

    typedef union
    {
        unsigned int u;
        float f;
    } _fp32_t;

    NeuralNetworkRuntime::Config config;

    vip_network network = nullptr;

    std::vector<vip_buffer_create_params_t> inputBufferParameters;
    std::vector<vip_buffer> inputBuffers;

    std::vector<vip_buffer_create_params_t> outputBufferParameters;
    std::vector<vip_buffer> outputBuffers;

    long frameCount = 0;
    
    static uint64_t get_perf_count()
    {
        struct timespec ts;

        clock_gettime(CLOCK_MONOTONIC, &ts);

        return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * 1000000000);
    }

    static void saveVectorToFile(const std::vector<int16_t>& vec, const std::string& filename) {
        std::ofstream outFile(filename);
        if (!outFile) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // outFile << std::fixed << std::setprecision(8); // 设置精度为8位小数

        for (const int16_t& val : vec) {
            outFile << val << "\n"; // 每个值占一行
        }

        outFile.close();
    }

    void queryBufferParameter(int index, vip_buffer_create_params_t &bufferCreateParams, BufferType type)
    {
        vip_status_e (*vipQueryBufferProp)(vip_network, vip_uint32_t, vip_enum, void *);

        if (type == BufferType::TYPE_IN)
        {
            vipQueryBufferProp = vip_query_input;
        }
        else if (type == BufferType::TYPE_OUT)
        {
            vipQueryBufferProp = vip_query_output;
        }
        else
        {
            throw std::invalid_argument("Invalid BufferType!");
        }

        memset(&bufferCreateParams, 0, sizeof(vip_buffer_create_params_t));

        vip_status_e status = vipQueryBufferProp(network, index, VIP_BUFFER_PROP_DATA_FORMAT, &bufferCreateParams.data_format);
        CHECK_VIP_STATUS(status);

        status = vipQueryBufferProp(network, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &bufferCreateParams.num_of_dims);
        CHECK_VIP_STATUS(status);

        status = vipQueryBufferProp(network, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, &bufferCreateParams.sizes);
        CHECK_VIP_STATUS(status);

        status = vipQueryBufferProp(network, index, VIP_BUFFER_PROP_QUANT_FORMAT, &bufferCreateParams.quant_format);
        CHECK_VIP_STATUS(status);

        switch (bufferCreateParams.quant_format)
        {
        case VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
            status = vipQueryBufferProp(network, index, VIP_BUFFER_PROP_FIXED_POINT_POS, &bufferCreateParams.quant_data.dfp.fixed_point_pos);
            CHECK_VIP_STATUS(status);
            break;
        case VIP_BUFFER_QUANTIZE_TF_ASYMM:
            status = vipQueryBufferProp(network, index, VIP_BUFFER_PROP_TF_SCALE, &bufferCreateParams.quant_data.affine.scale);
            CHECK_VIP_STATUS(status);

            status = vipQueryBufferProp(network, index, VIP_BUFFER_PROP_TF_ZERO_POINT, &bufferCreateParams.quant_data.affine.zeroPoint);
            CHECK_VIP_STATUS(status);
            break;
        case VIP_BUFFER_QUANTIZE_NONE:
        default:
            break;
        }
    }

    void createNeuralNetworkBuffers()
    {
        vip_uint32_t inputCount;
        vip_status_e status = vip_query_network(network, VIP_NETWORK_PROP_INPUT_COUNT, &inputCount);
        CHECK_VIP_STATUS(status);

        for (int i = 0; i < inputCount; i++)
        {
            vip_buffer_create_params_t bufferCreateParams;
            queryBufferParameter(i, bufferCreateParams, BufferType::TYPE_IN);

            inputBufferParameters.push_back(bufferCreateParams);

            vip_buffer inputBuffer;
            status = vip_create_buffer(&bufferCreateParams, sizeof(bufferCreateParams), &inputBuffer);
            CHECK_VIP_STATUS(status);

            inputBuffers.push_back(inputBuffer);
        }

        vip_uint32_t outputCount;
        status = vip_query_network(network, VIP_NETWORK_PROP_OUTPUT_COUNT, &outputCount);
        CHECK_VIP_STATUS(status);

        for (int i = 0; i < outputCount; i++)
        {
            vip_buffer_create_params_t bufferCreateParams;
            queryBufferParameter(i, bufferCreateParams, BufferType::TYPE_OUT);

            outputBufferParameters.push_back(bufferCreateParams);

            vip_buffer outputBuffer;
            status = vip_create_buffer(&bufferCreateParams, sizeof(bufferCreateParams), &outputBuffer);
            CHECK_VIP_STATUS(status);

            outputBuffers.push_back(outputBuffer);
        }
    }

    inline NeuralNetworkRuntime::InputDataFormat mapToInputDataFormat(vip_enum dataFormat)
    {
        NeuralNetworkRuntime::InputDataFormat format;

        switch (dataFormat)
        {
            case VIP_BUFFER_FORMAT_FP32:
                format = NeuralNetworkRuntime::InputDataFormat::FORMAT_FP32;
                break;
            case VIP_BUFFER_FORMAT_FP16:
                format = NeuralNetworkRuntime::InputDataFormat::FORMAT_FP16;
                break;
            case VIP_BUFFER_FORMAT_UINT8:
                format = NeuralNetworkRuntime::InputDataFormat::FORMAT_UINT8;
                break;
            case VIP_BUFFER_FORMAT_INT8:
                format = NeuralNetworkRuntime::InputDataFormat::FORMAT_INT8;
                break;
            case VIP_BUFFER_FORMAT_UINT16:
                format = NeuralNetworkRuntime::InputDataFormat::FORMAT_UINT16;
                break;
            case VIP_BUFFER_FORMAT_INT16:
                format = NeuralNetworkRuntime::InputDataFormat::FORMAT_INT16;
                break;
            default:
                format = NeuralNetworkRuntime::InputDataFormat::FORMAT_UNKNOWN;
                break;
        }

        return format;
    }

    void loadInputData(const NeuralNetworkRuntime::LoadingInputDataCallback& onLoadingInputData)
    {
        for(int i = 0; i < inputBuffers.size(); i++)
        {
            void *buffer = vip_map_buffer(inputBuffers[i]);

            vip_buffer_create_params_t &bufferCreateParams = inputBufferParameters[i];

            NeuralNetworkRuntime::InputDataFormat elementDataFormat = mapToInputDataFormat(bufferCreateParams.data_format);

            onLoadingInputData(i, buffer, elementDataFormat);

            vip_unmap_buffer(inputBuffers[i]);
        }
    }

    vip_uint32_t getFormatBytes(const vip_enum type)
    {
        switch (type)
        {
        case VIP_BUFFER_FORMAT_INT8:
        case VIP_BUFFER_FORMAT_UINT8:
            return 1;
        case VIP_BUFFER_FORMAT_INT16:
        case VIP_BUFFER_FORMAT_UINT16:
        case VIP_BUFFER_FORMAT_FP16:
        case VIP_BUFFER_FORMAT_BFP16:
            return 2;
        case VIP_BUFFER_FORMAT_FP32:
        case VIP_BUFFER_FORMAT_INT32:
        case VIP_BUFFER_FORMAT_UINT32:
            return 4;
        case VIP_BUFFER_FORMAT_FP64:
        case VIP_BUFFER_FORMAT_INT64:
        case VIP_BUFFER_FORMAT_UINT64:
            return 8;

        default:
            return 0;
        }
    }

    static float int8_to_fp32(signed char val, signed char fixedPointPos)
    {
        float result = 0.0f;

        if (fixedPointPos > 0)
        {
            result = (float)val * (1.0f / ((float)(1 << fixedPointPos)));
        }
        else
        {
            result = (float)val * ((float)(1 << -fixedPointPos));
        }

        return result;
    }

    static float int16_to_fp32(vip_int16_t val, signed char fixedPointPos)
    {
        float result = 0.0f;

        if (fixedPointPos > 0)
        {
            result = (float)val * (1.0f / ((float)(1 << fixedPointPos)));
        }
        else
        {
            result = (float)val * ((float)(1 << -fixedPointPos));
        }

        return result;
    }

    static vip_float_t affine_to_fp32(vip_int32_t val, vip_int32_t zeroPoint, vip_float_t scale)
    {
        vip_float_t result = 0.0f;
        result = ((vip_float_t)val - zeroPoint) * scale;
        return result;
    }

    static vip_float_t uint8_to_fp32(vip_uint8_t val, vip_int32_t zeroPoint, vip_float_t scale)
    {
        vip_float_t result = 0.0f;
        result = (val - (vip_uint8_t)zeroPoint) * scale;
        return result;
    }

    static float fp16_to_fp32(const short in)
    {
        const _fp32_t magic = {(254 - 15) << 23};
        const _fp32_t infnan = {(127 + 16) << 23};
        _fp32_t o;
        // Non-sign bits
        o.u = (in & 0x7fff) << 13;
        o.f *= magic.f;
        if (o.f >= infnan.f)
        {
            o.u |= 255 << 23;
        }
        // Sign bit
        o.u |= (in & 0x8000) << 16;
        return o.f;
    }

    vip_status_e integer_convert(const void *src, void *dest, vip_enum src_dtype, vip_enum dst_dtype)
    {
        vip_status_e status = VIP_SUCCESS;

        unsigned char all_zeros[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
        unsigned char all_ones[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        vip_uint32_t src_sz = getFormatBytes(src_dtype);
        vip_uint32_t dest_sz = getFormatBytes(dst_dtype);
        unsigned char *buffer = all_zeros;
        if (((vip_int8_t *)src)[src_sz - 1] & 0x80)
        {
            buffer = all_ones;
        }
        memcpy(buffer, src, src_sz);
        memcpy(dest, buffer, dest_sz);

        return status;
    }


    unsigned int get_file_size(const char *name)
    {
        FILE *fp = fopen(name, "rb");
        unsigned int size = 0;

        if (fp != NULL)
        {
            fseek(fp, 0, SEEK_END);
            size = ftell(fp);

            fclose(fp);
        }
        else
        {
            printf("Checking file %s failed.\n", name);
        }

        return size;
    }

    unsigned int load_file(const char *name, void *dst)
    {
        FILE *fp = fopen(name, "rb");
        unsigned int size = 0;

        if (fp != NULL)
        {
            fseek(fp, 0, SEEK_END);
            size = ftell(fp);

            fseek(fp, 0, SEEK_SET);
            size = fread(dst, size, 1, fp);

            fclose(fp);
        }

        return size;
    }

    std::vector<std::vector<float>> collectResults()
    {

        std::vector<vip_buffer>::size_type outputCount = outputBuffers.size();

        std::vector<std::vector<float>> results(outputCount);

        for (std::vector<vip_buffer>::size_type i = 0; i < outputCount; i++)
        {
            vip_buffer_create_params_t &bufferCreateParams = outputBufferParameters[i];

            vip_enum dataFormat = bufferCreateParams.data_format;
            vip_enum quantFormat = bufferCreateParams.quant_format;
            vip_int32_t fixed_point_pos = bufferCreateParams.quant_data.dfp.fixed_point_pos;
            vip_float_t scale = bufferCreateParams.quant_data.affine.scale;
            vip_int32_t zeroPoint = bufferCreateParams.quant_data.affine.zeroPoint;

            vip_uint32_t totalElementSize = 1;
            for (int j = 0; j < bufferCreateParams.num_of_dims; j++)
            {
                totalElementSize *= bufferCreateParams.sizes[j];
            }

            // vip_uint32_t formatSize = getFormatBytes(dataFormat);

            // std::cout << "dataFormat: " << dataFormat << std::endl;
            // std::cout << "quantFormat: " << quantFormat << std::endl;
            // std::cout << "fixed_point_pos: " << fixed_point_pos << std::endl;
            // std::cout << "scale: " << scale << std::endl;
            // std::cout << "zeroPoint: " << zeroPoint << std::endl;

            std::vector<float> result(totalElementSize);

            vip_uint8_t *buffer = (vip_uint8_t *)vip_map_buffer(outputBuffers[i]);

            std::vector<int16_t> shortBuffer(totalElementSize);
            std::memcpy(shortBuffer.data(), buffer, totalElementSize * sizeof(int16_t));

            float x;
            if (fixed_point_pos > 0)
            {
                x = 1.0f / ((float)(1 << fixed_point_pos));
            }
            else if (fixed_point_pos < 0)
            {
                x = (float)(1 << -fixed_point_pos);
            }
            else {
                x = 1;
            }
            
            for (vip_uint32_t j = 0; j < totalElementSize; j++)
            {
                result[j] = (float)shortBuffer[j] * x;
            }

            vip_unmap_buffer(outputBuffers[i]);

            results[i] = std::move(result);
        }

        return results;
    }

public:
    Impl(Config &config)
        : config(config)
    {

    }

    Impl(const Impl &other) = delete;
    Impl &operator=(const Impl &other) = delete;

    Impl(Impl &&other) noexcept
        : config(std::move(other.config)),
          network(other.network),
          inputBuffers(std::move(other.inputBuffers)),
          outputBufferParameters(std::move(other.outputBufferParameters)),
          outputBuffers(std::move(outputBuffers))
    {
        other.network = nullptr;
        for (auto &buffer : other.inputBuffers)
        {
            buffer = nullptr;
        }
        for (auto &buffer : other.outputBuffers)
        {
            buffer = nullptr;
        }
    }

    Impl &operator=(Impl &&other) noexcept
    {
        if (this != &other)
        {
            config = std::move(other.config);
            network = other.network;
            inputBuffers = std::move(other.inputBuffers);
            outputBufferParameters = std::move(other.outputBufferParameters);
            outputBuffers = std::move(other.outputBuffers);

            other.network = nullptr;
            for (auto &buffer : other.inputBuffers)
            {
                buffer = nullptr;
            }
            for (auto &buffer : other.outputBuffers)
            {
                buffer = nullptr;
            }
        }
        return *this;
    }

    void create()
    {
        std::cout << "NeuralNetworkRuntime Impl create..." << std::endl;

        //TODO thread-safe
        if (network != nullptr)
        {
            return;
        }

        try {
            vip_status_e status = vip_init(config.memSize);
            CHECK_VIP_STATUS(status);

            int file_size = get_file_size(config.modelFilePath.c_str());
            if (file_size <= 0)
            {
                printf("Network binary file %s can't be found.\n", config.modelFilePath.c_str());
                status = VIP_ERROR_INVALID_ARGUMENTS;
                CHECK_VIP_STATUS(status);
            }

            void *networkBuffer = malloc(file_size);
            load_file(config.modelFilePath.c_str(), networkBuffer);
            status = vip_create_network(networkBuffer, file_size, VIP_CREATE_NETWORK_FROM_MEMORY, &network);
            free(networkBuffer);
            CHECK_VIP_STATUS(status);

            createNeuralNetworkBuffers();

            std::cout << "vip_prepare_network start..." << std::endl;
            status = vip_prepare_network(network);
            std::cout << "vip_prepare_network finish" << std::endl;
            CHECK_VIP_STATUS(status);
        } 
        catch(std::exception &e)
        {
            destroy();
            throw e;
        }
    }

    std::vector<std::vector<float>> run(const NeuralNetworkRuntime::LoadingInputDataCallback& onLoadingInputData)
    {
        vip_status_e status = VIP_SUCCESS;

        loadInputData(onLoadingInputData);

        for (int i = 0; i < inputBuffers.size(); i++)
        {
            status = vip_set_input(network, i, inputBuffers[i]);
            CHECK_VIP_STATUS(status);
        }

        for (int i = 0; i < outputBuffers.size(); i++)
        {
            status = vip_set_output(network, i, outputBuffers[i]);
            CHECK_VIP_STATUS(status);
        }

        for (int i = 0; i < inputBuffers.size(); i++)
        {
            status = vip_flush_buffer(inputBuffers[i], VIP_BUFFER_OPER_TYPE_FLUSH);
            CHECK_VIP_STATUS(status);
        }

        status = vip_run_network(network);
        CHECK_VIP_STATUS(status);

        for (int i = 0; i < outputBuffers.size(); i++)
        {
            status = vip_flush_buffer(outputBuffers[i], VIP_BUFFER_OPER_TYPE_INVALIDATE);
            CHECK_VIP_STATUS(status);
        }

        // vip_inference_profile_t inferenceProfile;
        // status = vip_query_network(network, VIP_NETWORK_PROP_PROFILING, &inferenceProfile);
        // CHECK_VIP_STATUS(status);
        // std::cout << "-----inferenceTime = " << inferenceProfile.inference_time << std::endl;

        return collectResults();
    }

    void destroy()
    {
        if (network != nullptr)
        {
            vip_finish_network(network);

            vip_destroy_network(network);

            network = nullptr;

            for (int i = 0; i < inputBuffers.size(); i++)
            {
                vip_destroy_buffer(inputBuffers[i]);
                inputBuffers[i] = nullptr;
            }

            for (int i = 0; i < outputBuffers.size(); i++)
            {
                vip_destroy_buffer(outputBuffers[i]);
                outputBuffers[i] = nullptr;
            }

            vip_destroy();
        }
    }

    ~Impl()
    {
        destroy();
        std::cout << "NeuralNetworkRuntime destroyed!" << std::endl;
    }
};

NeuralNetworkRuntime::NeuralNetworkRuntime(Config &config)
    : _pImpl(new Impl(config))
{
    if (config.isAutoInit)
    {
        _pImpl->create();
    }
}

NeuralNetworkRuntime::NeuralNetworkRuntime(NeuralNetworkRuntime &&other) noexcept
    : _pImpl(std::move(other._pImpl))
{
    other._pImpl = nullptr;
}

NeuralNetworkRuntime &NeuralNetworkRuntime::operator=(NeuralNetworkRuntime &&other) noexcept
{
    if (this != &other)
    {
        _pImpl = std::move(other._pImpl);
        other._pImpl = nullptr;
    }
    return *this;
}

NeuralNetworkRuntime::~NeuralNetworkRuntime() = default;

void NeuralNetworkRuntime::create()
{
    _pImpl->create();
}

std::vector<std::vector<float>> NeuralNetworkRuntime::run(const LoadingInputDataCallback& onLoadingInputData)
{
    return _pImpl->run(onLoadingInputData);
}

void NeuralNetworkRuntime::destroy()
{
    _pImpl->destroy();
}
