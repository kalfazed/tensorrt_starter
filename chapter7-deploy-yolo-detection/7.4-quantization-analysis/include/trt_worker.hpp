#ifndef __WORKER_HPP__
#define __WORKER_HPP__

#include <memory>
#include <vector>
#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_classifier.hpp"
#include "trt_detector.hpp"

namespace thread{

class Worker {
public:
    Worker(std::string onnxPath, logger::Level level, model::Params params);
    void inference(std::string imagePath);

public:
    std::shared_ptr<logger::Logger>          m_logger;
    std::shared_ptr<model::Params>           m_params;

    std::shared_ptr<model::classifier::Classifier>  m_classifier;
    std::shared_ptr<model::detector::Detector>      m_detector;
    std::vector<float>                              m_scores;
    std::vector<model::detector::bbox>              m_boxes;
};

std::shared_ptr<Worker> create_worker(
    std::string onnxPath, logger::Level level, model::Params params);

}; //namespace thread

#endif //__WORKER_HPP__
