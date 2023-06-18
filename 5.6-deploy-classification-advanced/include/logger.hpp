#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include "NvInfer.h"
#include <string>
#include <stdarg.h>

#define LOGF(...) Logger::__log_info(Logger::Level::FATAL, __VA_ARGS__)
#define LOGE(...) Logger::__log_info(Logger::Level::ERROR, __VA_ARGS__)
#define LOGW(...) Logger::__log_info(Logger::Level::WARN, __VA_ARGS__)
#define LOG(...)  Logger::__log_info(Logger::Level::INFO, __VA_ARGS__)
#define LOGV(...) Logger::__log_info(Logger::Level::VERB, __VA_ARGS__)
#define LOGD(...) Logger::__log_info(Logger::Level::DEBUG, __VA_ARGS__)

#define DGREEN    "\033[1;36m"
#define BLUE      "\033[1;34m"
#define PURPLE    "\033[1;35m"
#define GREEN     "\033[1;32m"
#define YELLOW    "\033[1;33m"
#define RED       "\033[1;31m"
#define CLEAR     "\033[0m"



class Logger : public nvinfer1::ILogger{

public:
    enum class Level : int32_t{
        FATAL = 0,
        ERROR = 1,
        WARN  = 2,
        INFO  = 3,
        VERB  = 4,
        DEBUG = 5
    };

public:
    Logger();
    Logger(Level level);
    virtual void log(Severity severity, const char* msg) noexcept override;
    static void __log_info(Logger::Level level, const char* format, ...);
    Severity get_severity(Level level);
    Level get_level(Severity severity);

private:
    static Level m_level;
    Severity m_severity;
};

#endif //__LOGGER_HPP__
