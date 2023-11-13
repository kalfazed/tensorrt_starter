#ifndef __MODEL_HPP__
#define __MODEL_HPP__

// TensorRT related
#include "NvOnnxParser.h"
#include "NvInfer.h"

#include <memory>


class Model{

public:
    enum precision {
        FP32,
        FP16,
        INT8
    };

public:
    Model(std::string onnxPath, precision prec);
    bool build();
    bool infer(std::string imagePath);

private:
    bool build_from_onnx();
    bool preprocess();
    void print_network(nvinfer1::INetworkDefinition &network, bool optimized);

private:
    std::string mOnnxPath = "";
    std::string mEnginePath = "";
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    float* mInputHost;
    float* mInputDevice;
    float* mOutputHost;
    float* mOutputDevice;
    int mInputSize;
    int mOutputSize;
    nvinfer1::DataType mPrecision;
};

#endif // __MODEL_HPP__
