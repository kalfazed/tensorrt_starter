#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    /*这么实现目的在于让调用的整个过程精简化*/
    string onnxPath    = "models/onnx/yolov8n.onnx";

    auto level         = logger::Level::VERB;
    auto params        = model::Params();

    params.img         = {640, 640, 3};
    params.task        = model::task_type::DETECTION;
    params.dev         = model::device::GPU;
    params.prec        = model::precision::INT8;

    // 创建一个worker的实例, 在创建的时候就完成初始化
    auto worker   = thread::create_worker(onnxPath, level, params);

    // 根据worker中的task类型进行推理
    worker->inference("data/source/car.jpg");
    worker->inference("data/source/bedroom.jpg");
    worker->inference("data/source/crossroad.jpg");
    worker->inference("data/source/airport.jpg");

    return 0;

    /*
     * 这里记录一下测试结果
     * TensorRT 8.6
     *  - IInt8EntropyCalibrator2: int8精度下降严重，classness掉点严重
     *  - IInt8MinMaxCalibrator:   int8推理精度可以恢复, classness的max会凸显出来
     * 
     * 为什么呢？
     * 其实只要我们回顾一下因为MinMax会把FP32中的最大最小值也会留下来
     * 但为什么yolov8的fp32的最大最小值会如此重要呢？因为C2F架构中会DWConv
     * depthwise的存在会潜在性的让output tensor的FP32在每一个layer都会有很大不同的分布
     * 如果用entropy的话，很有可能会让某些关键信息流失掉
     *
     * 注意，当我们换了模型(e.g. yolov8n -> yolov8x)，
     * 或者换了calibrator(e.g. IInt8EntropyCalibrator2 -> IInt8MinMaxCalibrator)以后
     * 我们必须要把calibration_table给删除掉重新制作。
     * 因为之前的calibration_table记载的层的dynamic range的统计信息无法被复用，会报错
     *
    */
}

