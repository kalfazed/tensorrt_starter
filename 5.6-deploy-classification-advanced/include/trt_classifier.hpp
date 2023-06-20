#ifndef __TRT_CLASSIFIER_HPP__
#define __TRT_CLASSIFIER_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "logger.hpp"
#include "trt_model.hpp"

namespace classifier {
class Classifier : public Model{

public:
    Classifier(std::string onnx_path, Logger::Level level, Params params) : Model(onnx_path, level, params) {};

public:
    virtual void setup(nvinfer1::IRuntime& runtime, void const* data, std::size_t size) override;
    virtual bool preprocess_cpu() override;
    virtual bool preprocess_gpu() override;
    virtual bool postprocess_cpu() override;
    virtual bool postprocess_gpu() override;

private:
    float m_confidence;
    std::string m_label;
    int m_inputSize; 
    int m_outputSize;
};

std::shared_ptr<Classifier> make_classifier(
    std::string onnx_path, Logger::Level level, Model::Params params);

}; //namespace classifier

#endif //__TRT_CLASSIFIER_HPP__
