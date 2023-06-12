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
    if (fileExists("models/example_two_head.engine")){
        cout << "trt engine has been generated!" << endl;
        return true;
    }
    Logger logger;
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    config->setMaxWorkspaceSize(1<<28);

    if (!parser->parseFromFile("models/example_two_head.onnx", 1)){
        return false;
    }

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    auto f = fopen("models/example_two_head.engine", "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    mInputDims         = network->getInput(0)->getDimensions();
    mOutputDims        = network->getOutput(0)->getDimensions();
    return true;
};

bool Model::infer(){

    /* 1. 读取model => 创建runtime, engine, context */
    string planFilePath = "models/example_two_head.engine";
    if (!fileExists(planFilePath)) {
        return false;
    }

    vector<unsigned char> modelData;
    modelData = loadFile(planFilePath);
    
    Logger logger;
    auto runtime     = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine      = make_unique<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    auto context     = make_unique<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    auto input_dims0   = context->getBindingDimensions(0);
    auto output_dims0  = context->getBindingDimensions(1);
    auto output_dims1  = context->getBindingDimensions(2);

    cout << "input0 dim shape is:  " << printDims(input_dims0) << endl;
    cout << "output0 dim shape is: " << printDims(output_dims0) << endl;
    cout << "output1 dim shape is: " << printDims(output_dims1) << endl;

    /* 2. host->device的数据传递 */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* host memory上的数据*/
    float input_host[]    = {1, 2, 3, 4};
    float output_host[3];

    /* device memory上的数据*/
    float* input_device = nullptr;
    float* weight_device = nullptr;
    float* output_device = nullptr;

    int input_size = 4;
    int output_size = 3;

    /* 分配空间, 并传送数据从host到device*/
    cudaMalloc(&input_device, sizeof(input_host));
    cudaMalloc(&output_device, sizeof(output_host));
    cudaMemcpyAsync(input_device, input_host, sizeof(input_host), cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

    /* 3. 模型推理, 最后做同步处理 */
    float* bindings[] = {input_device, output_device};
    bool success = context->enqueueV2((void**)bindings, stream, nullptr);

    /* 4. device->host的数据传递 */
    cudaMemcpyAsync(output_host, output_device, sizeof(output_host), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    cout << "input data is: " << printTensor(input_host, input_size) << endl;
    cout << "output data is:" << printTensor(output_host, output_size) << endl;
    cout << "finished inference" << endl;

    cudaStreamDestroy(stream);
    return true;
}


