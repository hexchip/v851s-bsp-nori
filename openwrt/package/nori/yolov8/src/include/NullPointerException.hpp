#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

#define CHECK_NULL_PTR(ptr)                                               \
    do                                                                    \
    {                                                                     \
        if (nullptr == ptr)                                               \
        {                                                                 \
            throw NullPointerException(__FILE__, __FUNCTION__, __LINE__); \
        }                                                                 \
    } while (0)

class NullPointerException : public std::runtime_error
{
public:
    NullPointerException(const std::string &file, const std::string &function, int line)
        : std::runtime_error(buildErrorMessage(file, function, line)) {}

private:
    static std::string buildErrorMessage(const std::string &file, const std::string &function, int line)
    {
        std::ostringstream oss;
        oss << "Null pointer exception occurred in " << function << " at " << file << " line " << line;
        return oss.str();
    }
};