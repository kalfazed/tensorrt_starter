#include <memory>
#include <iostream>
#include <string>
#include <type_traits>

#include "model.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "utils.hpp"
#include "cuda_runtime.h"

using namespace std;

//自己创建的logger需要继承ILogger,并实现log虚函数
class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char* msg) noexcept override{
        string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = RED    "[fatal]:" CLEAR;
            case Severity::kERROR:          str = RED    "[error]:" CLEAR;
            case Severity::kWARNING:        str = BLUE   "[warn]:"  CLEAR;
            case Severity::kINFO:           str = YELLOW "[info]:"  CLEAR;
            case Severity::kVERBOSE:        str = PURPLE "[verb]:"  CLEAR;
        }
        if (severity <= Severity::kINFO)
            cout << str << string(msg) << endl;
    }
};


struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using make_unique = std::unique_ptr<T, InferDeleter>;

Model::Model(string onnxPath){
    if (!fileExists(onnxPath)) {
        LOGE("%s not found. Program terminated", onnxPath.c_str());
        exit(1);
    }
    mOnnxPath   = onnxPath;
    mEnginePath = getEnginePath(mOnnxPath);
}

bool Model::build(){
    if (fileExists(mEnginePath)){
        LOG("%s has been generated!", mEnginePath.c_str());
        return true;
    } else {
        LOG("%s not found. Building engine...", mEnginePath.c_str());
    }
    Logger logger;
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    config->setMaxWorkspaceSize(1<<28);

    if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
        LOGE("ERROR: failed to %s", mOnnxPath.c_str());
        return false;
    }

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    auto f = fopen(mEnginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    mInputDims         = network->getInput(0)->getDimensions();
    mOutputDims        = network->getOutput(0)->getDimensions();
    LOG("Input dim is %s", printDims(mInputDims).c_str());
    LOG("Output dim is %s", printDims(mOutputDims).c_str());
    return true;
};
