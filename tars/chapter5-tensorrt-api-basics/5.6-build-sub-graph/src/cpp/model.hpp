#ifndef __MODEL_HPP__
#define __MODEL_HPP__

// TensorRT related
#include "NvOnnxParser.h"
#include "NvInfer.h"

#include <string>
#include <map>
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
    bool infer();

private:
    void init_data(nvinfer1::Dims input_dims, nvinfer1::Dims output_dims);
    bool build_from_onnx();
    bool build_from_weights();

    bool constructNetwork();
    bool preprocess();
    void print_network(nvinfer1::INetworkDefinition &network, bool optimized);
    std::map<std::string, nvinfer1::Weights> loadWeights();

private:
    std::string mWtsPath = "";
    std::string mOnnxPath = "";
    std::string mEnginePath = "";
    std::map<std::string, nvinfer1::Weights> mWts;
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
