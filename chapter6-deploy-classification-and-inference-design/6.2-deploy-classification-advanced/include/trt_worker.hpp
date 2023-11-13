#ifndef __WORKER_HPP__
#define __WORKER_HPP__

#include <memory>
#include <vector>
#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_classifier.hpp"

namespace thread{

class Worker {
public:
    Worker(std::string onnxPath, logger::Level level, model::Params params);
    void inference(std::string imagePath);

public:
    std::shared_ptr<logger::Logger>          m_logger;
    std::shared_ptr<model::Params>           m_params;

    std::shared_ptr<model::classifier::Classifier>  m_classifier; // 因为今后考虑扩充为multi-task，所以各个task都是worker的成员变量
    std::vector<float>                       m_scores;     // 因为今后考虑会将各个multi-task间进行互动，所以worker需要保存各个task的结果
};

std::shared_ptr<Worker> create_worker(
    std::string onnxPath, logger::Level level, model::Params params);

}; //namespace thread

#endif //__WORKER_HPP__
