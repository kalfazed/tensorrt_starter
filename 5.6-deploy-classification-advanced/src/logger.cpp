#include "logger.hpp"
#include "NvInfer.h"

using namespace std;

Logger::Level Logger::m_level = Level::INFO;

Logger::Logger(Logger::Level level) {
    m_level = level;
    m_severity = get_severity(Logger::Level::WARN);
    // m_severity = get_severity(level);
}

Logger::Severity Logger::get_severity(Logger::Level level) {
    switch (level) {
        case Level::FATAL: return Severity::kINTERNAL_ERROR;
        case Level::ERROR: return Severity::kERROR;
        case Level::WARN:  return Severity::kWARNING;
        case Level::INFO:  return Severity::kINFO;
        case Level::VERB:  return Severity::kVERBOSE;
        default:           return Severity::kVERBOSE;
    }
}

Logger::Level Logger::get_level(Severity severity) {
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
    if (severity <= m_severity) __log_info(get_level(severity), "%s", msg);
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
        exit(1);
    }
}
