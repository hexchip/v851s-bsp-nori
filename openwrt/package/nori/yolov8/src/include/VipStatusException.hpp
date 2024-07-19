#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

#include "vip_lite.h"

#define CHECK_VIP_STATUS(status)                                                    \
    do                                                                              \
    {                                                                               \
        if (VIP_SUCCESS != status)                                                  \
        {                                                                           \
            throw VipStatusException(status, "", __FILE__, __FUNCTION__, __LINE__); \
        }                                                                           \
    } while (0)

#define CHECK_VIP_STATUS_WITH_MSG(status, format, args...)                                  \
    do                                                                                      \
    {                                                                                       \
        if (VIP_SUCCESS != status)                                                          \
        {                                                                                   \
            char msg_buffer[256];                                                           \
            snprintf(msg_buffer, sizeof(msg_buffer), format, args);                         \
            throw VipStatusException(status, msg_buffer, __FILE__, __FUNCTION__, __LINE__); \
        }                                                                                   \
    } while (0)

class VipStatusException : public std::runtime_error
{
public:
    VipStatusException(vip_status_e status, const std::string &msg, const std::string &file, const std::string &function, int line)
        : std::runtime_error(buildErrorMessage(status, msg, file, function, line)) {}

private:
    static std::string buildErrorMessage(vip_status_e status, const std::string &msg, const std::string &file, const std::string &function, int line)
    {
        std::ostringstream oss;
        oss << "vip status: " << status << "\n";
        if (!msg.empty())
        {
            oss << msg << "\n";
        }
        oss << "exception occurred in " << function << " at " << file << " line " << line;
        return oss.str();
    }
};