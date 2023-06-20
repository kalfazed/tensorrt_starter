#include "worker.hpp"
#include "trt_classifier.hpp"
#include "memory"

using namespace std;

namespace thread{

Worker::Worker(string onnxPath, Model::task_type type, Logger::Level level, Model::Params params) {
    // 这里根据task_type选择创建的trt_model的子类，今后会针对detection, segmentation扩充
    if (type == Model::task_type::CLASSIFICATION) 
        m_classifier = classifier::make_classifier(onnxPath, level, params);
}

void Worker::trt_infer(string imagePath) {
    if (m_classifier != nullptr) {
        m_classifier->init_model();
        m_classifier->load_image(imagePath);
        m_classifier->inference();
    }
}

shared_ptr<Worker> create_worker(
    std::string onnxPath, Model::task_type type, Logger::Level level, Model::Params params) 
{
    auto worker = make_shared<Worker>(onnxPath, type, level, params);
    return worker;
}

}; // namespace thread
