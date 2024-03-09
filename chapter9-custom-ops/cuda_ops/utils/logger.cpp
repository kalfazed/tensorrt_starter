#include "logger.hpp"
#include "NvInfer.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <cstdlib>

using namespace std;

namespace logger {

Level Logger::m_level = Level::kINFO;

Logger::Logger(Level level) {
    m_level = level;
    m_severity = get_severity(level);
}

// @brief Get current time
std::string TimeNow() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%H:%M:%S");
  return ss.str();
}

Logger::Severity Logger::get_severity(Level level) {
    switch (level) {
        case Level::kFATAL: return Severity::kINTERNAL_ERROR;
        case Level::kERROR: return Severity::kERROR;
        case Level::kWARN:  return Severity::kWARNING;
        case Level::kINFO:  return Severity::kINFO;
        case Level::kVERB:  return Severity::kVERBOSE;
        default:           return Severity::kVERBOSE;
    }
}

Level Logger::get_level(Severity severity) {
    string str;
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return Level::kFATAL;
        case Severity::kERROR:          return Level::kERROR;
        case Severity::kWARNING:        return Level::kWARN;
        case Severity::kINFO:           return Level::kINFO;
        case Severity::kVERBOSE:        return Level::kVERB;
    }
}

void Logger::log (Severity severity, const char* msg) noexcept{
    /* 
        有的时候TensorRT给出的log会比较多并且比较细，所以我们选择将TensorRT的打印log的级别稍微约束一下
        - TensorRT的log级别如果是FATAL, ERROR, WARNING, 按照正常方式打印
        - TensorRT的log级别如果是INFO或者是VERBOSE的时候，只有当logger的level在大于VERBOSE的时候再打出
    */
    if (severity <= get_severity(Level::kWARN)
        || m_level >= Level::kDEBUG)
        __log_info(get_level(severity), "%s", msg);
}

void Logger::__log_info(Level level, const char* format, ...) {
  std::array<char, MAXLOGSIZE> msg;

  va_list args;
  va_start(args, format);
  int n = 0;
  auto now = TimeNow();

  // print time
  n += snprintf(msg.data() + n, sizeof(msg) - n, BLUE "(%s) " CLEAR,
                now.c_str());

  // print log level
  switch (level) {
    case Level::kDEBUG:
      n += snprintf(msg.data() + n, sizeof(msg) - n, DGREEN "[DEBUG]: " CLEAR);
      break;
    case Level::kVERB:
      n += snprintf(msg.data() + n, sizeof(msg) - n, PURPLE "[VERB]: " CLEAR);
      break;
    case Level::kINFO:
      n += snprintf(msg.data() + n, sizeof(msg) - n, GREEN "[INFO]: " CLEAR);
      break;
    case Level::kWARN:
      n += snprintf(msg.data() + n, sizeof(msg) - n, YELLOW "[WARN]: " CLEAR);
      break;
    case Level::kERROR:
      n += snprintf(msg.data() + n, sizeof(msg) - n, RED "[ERROR]: " CLEAR);
      break;
    default:
      n += snprintf(msg.data() + n, sizeof(msg) - n, RED "[FATAL]: " CLEAR);
      break;
  }

  // print va_args
  n += vsnprintf(msg.data() + n, sizeof(msg) - n, format, args);

  va_end(args);

  if (level <= m_level) {
    fprintf(stdout, "%s\n", msg.data());
  }

  if (level <= Level::kERROR) {
    fflush(stdout);
  }
}

shared_ptr<Logger> create_logger(Level level) {
    return make_shared<Logger>(level);
}

} // namespace logger
