#include "trt_model.hpp"
#include "logger.hpp"
#include "worker.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    /*这么实现目的在于让调用的整个过程精简化*/
    auto onnxPath = "models/resnet18.onnx";
    auto test_img = "data/tiny-cat.png";

    auto level    = Logger::Level::INFO;
    auto task     = Model::task_type::CLASSIFICATION;
    auto device   = Model::device::GPU;
    auto params   = Model::Params(224, 224, 3, 1000, Model::device::GPU);

    // 创建一个worker的实例, 在创建的时候就完成初始化
    auto worker   = thread::create_worker(onnxPath, task, level, params);

    // 根据worker中的task类型进行推理
    worker->trt_infer(test_img);

    return 0;
}
