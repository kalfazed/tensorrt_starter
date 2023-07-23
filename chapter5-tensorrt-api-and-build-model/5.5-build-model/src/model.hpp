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
    Model(std::string onnxPath);
    bool build();
    bool infer();

private:
    void init_data();
    bool build_from_onnx();
    bool build_from_weights();

    void build_linear(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_conv(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_permute(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_reshape(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_batchNorm(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_cbr(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);

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
};

#endif // __MODEL_HPP__
