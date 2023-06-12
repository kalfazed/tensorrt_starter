#include "trt_model.hpp"
#include "utils.hpp"
#include "logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <string>

using namespace std;

class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char* msg) noexcept override{
        string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = RED    "[fatal]" CLEAR;
            case Severity::kERROR:          str = RED    "[error]" CLEAR;
            case Severity::kWARNING:        str = BLUE   "[warn]"  CLEAR;
            case Severity::kINFO:           str = GREEN  "[info]"  CLEAR;
            case Severity::kVERBOSE:        str = PURPLE "[verb]"  CLEAR;
        }

        if (severity <= Severity::kINFO)
            LOG("%s:%s", str.c_str(), msg);
    }
};

Model::Model(string onnx_path, task_type type) {
    m_onnxPath = onnx_path;
    m_enginePath = getEnginePath(m_onnxPath);

    if (!fileExists(m_enginePath)){
        LOG("trt engine not found, building trt engine...");
        build_engine();
    } else {
        LOG("trt engine has been generated! continue...");
    }
}

bool Model::build_engine() {
    
    Logger logger;
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    config->setMaxWorkspaceSize(m_workspaceSize);

    if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){
        return false;
    }

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    auto f = fopen(m_enginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    m_engine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    m_inputDims         = network->getInput(0)->getDimensions();
    m_outputDims        = network->getOutput(0)->getDimensions();
    return true;
}


