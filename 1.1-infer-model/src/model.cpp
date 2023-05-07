#include "model.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char* msg) noexcept override{
        string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = "[fatal]";
            case Severity::kERROR:          str = "[error]";
            case Severity::kWARNING:        str = "[warn]";
            case Severity::kINFO:           str = "[info]";
            case Severity::kVERBOSE:        str = "[verb]";
        }

        if (severity <= Severity::kINFO)
            cout << str << ":" << string(msg) << endl;
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

bool Model::build(){
    Logger logger;
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    config->setMaxWorkspaceSize(1<<28);

    if (!parser->parseFromFile("models/mnist.onnx", 1)){
        return false;
    }

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    auto f = fopen("models/mnist.engine", "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    mInputDims         = network->getInput(0)->getDimensions();
    mOutputDims        = network->getOutput(0)->getDimensions();
    return true;
};

void Model::fileRead(const string& path, void* memory, int& size){
    stringstream sModel;
    ifstream cache(path);

    /* 将engine的内容写入sModel中*/
    sModel.seekg(0, sModel.beg);
    sModel << cache.rdbuf();
    cache.close();

    /* 计算model的大小*/
    sModel.seekg(0, ios::end);
    size = sModel.tellg();
    sModel.seekg(0, ios::beg);
    memory = malloc(size);

    /* 将sModel中的stream通过read函数写入modelMem中*/
    sModel.read((char*)memory, size);
}


bool Model::infer(){
    /*
        我们在infer需要做的事情
        1. 读取model => 创建runtime, engine, context
        2. 预处理 (读取数据，创建buff存储数据，并且将数据从0~255缩放到0~1)
        3. 把数据进行host->device传输
        4. 使用context推理
        5. 把数据进行device->host传输
        6. 后处理 (计算softmax，获取最大值并打印)
    */

    // 1. 读取model => 创建runtime, engine, context
    string planFilePath = "models/mnist.engine";
    void* modelMemory;
    int modelSize;
    fileRead(planFilePath, modelMemory, modelSize);
    
    Logger logger;
    auto runtime = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine  = make_unique<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelMemory, modelSize));
    auto context = make_unique<nvinfer1::IExecutionContext>(engine->createExecutionContext());


    // 2. 预处理 (读取数据，创建buff存储数据，并且将数据从0~255缩放到0~1)
    // 3. 把数据进行host->device传输
    // 4. 使用context推理
    // 5. 把数据进行device->host传输
    // 6. 后处理 (计算softmax，获取最大值并打印)

   return true;
}


