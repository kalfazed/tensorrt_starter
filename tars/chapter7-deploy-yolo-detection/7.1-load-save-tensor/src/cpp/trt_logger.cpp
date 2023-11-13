#include "trt_logger.hpp"
#include "NvInfer.h"
#include <cstdlib>

using namespace std;

namespace logger {

Level Logger::m_level = Level::INFO;

Logger::Logger(Level level) {
    m_level = level;
    m_severity = get_severity(level);
}

Logger::Severity Logger::get_severity(Level level) {
    switch (level) {
        case Level::FATAL: return Severity::kINTERNAL_ERROR;
        case Level::ERROR: return Severity::kERROR;
        case Level::WARN:  return Severity::kWARNING;
        case Level::INFO:  return Severity::kINFO;
        case Level::VERB:  return Severity::kVERBOSE;
        default:           return Severity::kVERBOSE;
    }
}

Level Logger::get_level(Severity severity) {
    string str;
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return Level::FATAL;
        case Severity::kERROR:          return Level::ERROR;
        case Severity::kWARNING:        return Level::WARN;
        case Severity::kINFO:           return Level::INFO;
        case Severity::kVERBOSE:        return Level::VERB;
    }
}

void Logger::log (Severity severity, const char* msg) noexcept{
    /* 
        有的时候TensorRT给出的log会比较多并且比较细，所以我们选择将TensorRT的打印log的级别稍微约束一下
        - TensorRT的log级别如果是FATAL, ERROR, WARNING, 按照正常方式打印
        - TensorRT的log级别如果是INFO或者是VERBOSE的时候，只有当logger的level在大于VERBOSE的时候再打出
    */
    if (severity <= get_severity(Level::WARN)
        || m_level >= Level::DEBUG)
        __log_info(get_level(severity), "%s", msg);
}

void Logger::__log_info(Level level, const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);
    int n = 0;
    
    switch (level) {
        case Level::DEBUG: n += snprintf(msg + n, sizeof(msg) - n, DGREEN "[debug]" CLEAR); break;
        case Level::VERB:  n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[verb]" CLEAR); break;
        case Level::INFO:  n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[info]" CLEAR); break;
        case Level::WARN:  n += snprintf(msg + n, sizeof(msg) - n, BLUE "[warn]" CLEAR); break;
        case Level::ERROR: n += snprintf(msg + n, sizeof(msg) - n, RED "[error]" CLEAR); break;
        default:           n += snprintf(msg + n, sizeof(msg) - n, RED "[fatal]" CLEAR); break;
    }

    n += vsnprintf(msg + n, sizeof(msg) - n, format, args);

    va_end(args);

    if (level <= m_level) 
        fprintf(stdout, "%s\n", msg);

    if (level <= Level::ERROR) {
        fflush(stdout);
        exit(0);
    }
}

shared_ptr<Logger> create_logger(Level level) {
    return make_shared<Logger>(level);
}

} // namespace logger
