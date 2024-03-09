#pragma once

#include "NvInfer.h"
#include <string>
#include <stdarg.h>
#include <memory>

#define LOGF(...) logger::Logger::__log_info(logger::Level::kFATAL, __VA_ARGS__)
#define LOGE(...) logger::Logger::__log_info(logger::Level::kERROR, __VA_ARGS__)
#define LOGW(...) logger::Logger::__log_info(logger::Level::kWARN,  __VA_ARGS__)
#define LOG(...)  logger::Logger::__log_info(logger::Level::kINFO,  __VA_ARGS__)
#define LOGV(...) logger::Logger::__log_info(logger::Level::kVERB,  __VA_ARGS__)
#define LOGD(...) logger::Logger::__log_info(logger::Level::kDEBUG, __VA_ARGS__)

#define DGREEN    "\033[1;36m"
#define BLUE      "\033[1;34m"
#define PURPLE    "\033[1;35m"
#define GREEN     "\033[1;32m"
#define YELLOW    "\033[1;33m"
#define RED       "\033[1;31m"
#define CLEAR     "\033[0m"

#define MAXLOGSIZE 1000

namespace logger{

enum class Level : int32_t{
    kFATAL = 0,
    kERROR = 1,
    kWARN  = 2,
    kINFO  = 3,
    kVERB  = 4,
    kDEBUG = 5
};

class Logger : public nvinfer1::ILogger{

public:
    Logger();
    Logger(Level level);
    virtual void log(Severity severity, const char* msg) noexcept override;
    static void __log_info(Level level, const char* format, ...);
    Severity get_severity(Level level);
    Level get_level(Severity severity);

private:
    static Level m_level;
    Severity m_severity;
};

std::shared_ptr<Logger> create_logger(Level level);

} // namespace logger
