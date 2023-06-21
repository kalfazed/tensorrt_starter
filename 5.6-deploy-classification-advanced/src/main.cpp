#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_worker.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    /*这么实现目的在于让调用的整个过程精简化*/
    auto onnxPath      = "models/resnet18.onnx";
    auto test_img      = "data/tiny-cat.png";

    auto level         = logger::Level::INFO;
    auto params        = model::Params();

    params.img         = {224, 224, 3};
    params.num_cls     = 1000;
    params.task        = model::task_type::CLASSIFICATION;
    params.dev         = model::device::GPU;

    // 创建一个worker的实例, 在创建的时候就完成初始化
    auto worker   = thread::create_worker(onnxPath, level, params);

    // 根据worker中的task类型进行推理
    worker->inference(test_img);

    return 0;
}
