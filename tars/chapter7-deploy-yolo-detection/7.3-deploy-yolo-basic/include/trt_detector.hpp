#ifndef __TRT_DETECTOR_HPP__
#define __TRT_DETECTOR_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_logger.hpp"
#include "trt_model.hpp"

namespace model{

namespace detector {

enum model {
    YOLOV5,
    YOLOV8
};

struct bbox {
    float x0, x1, y0, y1;
    float confidence;
    bool  flg_remove;
    int   label;
    
    bbox() = default;
    bbox(float x0, float y0, float x1, float y1, float conf, int label) : 
        x0(x0), y0(y0), x1(x1), y1(y1), 
        confidence(conf), flg_remove(false), 
        label(label){};
};

class Detector : public Model{

public:
    // 这个构造函数实际上调用的是父类的Model的构造函数
    Detector(std::string onnx_path, logger::Level level, Params params) : 
        Model(onnx_path, level, params) {};

public:
    // 这里detection自己实现了一套前处理/后处理，以及内存分配的初始化
    virtual void setup(void const* data, std::size_t size) override;
    virtual void reset_task() override;
    virtual bool preprocess_cpu() override;
    virtual bool preprocess_gpu() override;
    virtual bool postprocess_cpu() override;
    virtual bool postprocess_gpu() override;

private:
    std::vector<bbox> m_bboxes;
    int m_inputSize; 
    int m_imgArea;
    int m_outputSize;
};

// 外部调用的接口
std::shared_ptr<Detector> make_detector(
    std::string onnx_path, logger::Level level, Params params);

}; // namespace detector
}; // namespace model

#endif //__TRT_DETECTOR_HPP__
