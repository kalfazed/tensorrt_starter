#ifndef __MODEL_HPP__
#define __MODEL_HPP__

// TensorRT related
#include "NvOnnxParser.h"
#include "NvInfer.h"

#include <memory>
#include <string>


class Model{
public:
    Model(std::string onnxPath);
    bool build();
    bool infer(std::string imagePath);
private:
    std::string m_onnxPath;
    std::string m_enginePath;
    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    bool constructNetwork();
    bool preprocess();
};

#endif // __MODEL_HPP__
