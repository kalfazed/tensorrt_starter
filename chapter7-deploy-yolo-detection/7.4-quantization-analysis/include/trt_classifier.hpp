#ifndef __TRT_CLASSIFIER_HPP__
#define __TRT_CLASSIFIER_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_logger.hpp"
#include "trt_model.hpp"

namespace model{

namespace classifier {
class Classifier : public Model{

public:
    // 这个构造函数实际上调用的是父类的Model的构造函数
    Classifier(std::string onnx_path, logger::Level level, Params params) : 
        Model(onnx_path, level, params) {};

public:
    // 这里classifer自己实现了一套前处理/后处理，以及内存分配的初始化
    virtual void setup(void const* data, std::size_t size) override;
    virtual void reset_task() override;
    virtual bool preprocess_cpu() override;
    virtual bool preprocess_gpu() override;
    virtual bool postprocess_cpu() override;
    virtual bool postprocess_gpu() override;

private:
    float m_confidence;
    std::string m_label;
    int m_inputSize; 
    int m_imgArea;
    int m_outputSize;
};

// 外部调用的接口
std::shared_ptr<Classifier> make_classifier(
    std::string onnx_path, logger::Level level, Params params);

}; // namespace classifier
}; // namespace model

#endif //__TRT_CLASSIFIER_HPP__
