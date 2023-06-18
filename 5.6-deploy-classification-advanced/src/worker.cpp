#include "worker.hpp"
#include "memory"

using namespace std;

Worker::Worker(string onnxPath, Model::task_type type, Logger::Level level, Model::Params params) {
    m_logger = make_shared<Logger>(level);
    m_classifier = make_shared<Model>(onnxPath, type, level, params);
    m_params = make_shared<Model::Params>(params);
}

void Worker::trt_infer(string imagePath) {
    m_classifier->load_image(imagePath);
    m_classifier->infer_classifier();
}
