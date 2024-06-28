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
    void init_data(nvinfer1::Dims, nvinfer1::Dims);
    bool build_from_onnx();
    bool build_from_weights();

    /* 新添加的小模型 */
    void build_linear(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_conv(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_permute(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_reshape(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_batchNorm(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_cbr(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_pooling(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_upsample(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_deconv(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_concat(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts, int concatDim = 1);
    void build_elementwise(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_reduce(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
    void build_slice(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts, int sliceDim = 1, int sliceIndex = 0);

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
