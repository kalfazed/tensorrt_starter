#include <iostream>
#include <memory>

#include "trt_model.hpp"
#include "utils.hpp"
#include "logger.hpp"
#include "worker.hpp"
#include "memory"

using namespace std;

int main(int argc, char const *argv[])
{
    auto level    = Logger::Level::VERB;
    auto onnxPath = "models/resnet18.onnx";
    auto test_img = "data/tiny-cat.png";
    auto mode     = Model::task_type::CLASSIFICATION;
    auto params   = Model::Params(224, 224, 3, 1000);

    auto worker = make_shared<Worker>(onnxPath, mode, level, params);
    worker->m_classifier->init_model();
    worker->m_classifier->load_image("data/tiny-cat.png");
    worker->m_classifier->infer_classifier();

    return 0;
}
