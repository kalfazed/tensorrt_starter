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

Model::Model(string onnxPath){
    this->mOnnxPath = onnxPath;
    this->mEnginePath = getEnginePath(mOnnxPath);
}

bool Model::build(){
    if (fileExists(mEnginePath)){
        cout << "trt engine has been generated!" << endl;
    }

    Logger logger;
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    config->setMaxWorkspaceSize(1<<28);

    if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
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

    // print layer info
    int inputCount = network->getNbInputs();
    int outputCount = network->getNbOutputs();
    string layer_info;

    for (int i = 0; i < inputCount; i++) {
        layer_info = "";
        cout << "Input info: ";
        auto input = network->getInput(i);
        layer_info  += input->getName();
        layer_info  += ": ";
        layer_info  += printTensorShape(input);
        cout << layer_info << endl;
    }

    for (int i = 0; i < outputCount; i++) {
        layer_info = "";
        cout << "Output info: ";
        auto output = network->getOutput(i);
        layer_info  += output->getName();
        layer_info  += ": ";
        layer_info  += printTensorShape(output);
        cout << layer_info << endl;
    }

    int layerCount = network->getNbLayers();
    printf("network has %d layers\n", layerCount);
    for (int i = 0; i < layerCount; i++) {
        char layer_info[1000];
        auto layer   = network->getLayer(i);
        auto input   = layer->getInput(0);
        int n = 0;
        if (input == nullptr){
            continue;
        }
        auto output  = layer->getOutput(0);

        n += sprintf(layer_info + n, "layer_info:  ");
        n += sprintf(layer_info + n, "%-40s:", layer->getName());
        n += sprintf(layer_info + n, "%-25s", printTensorShape(input).c_str());
        n += sprintf(layer_info + n, " -> ");
        n += sprintf(layer_info + n, "%-25s", printTensorShape(output).c_str());
        n += sprintf(layer_info + n, "[%s]", getPrecision(layer->getPrecision()).c_str());
        cout << layer_info << endl;
    }
    return true;
};

bool Model::infer(){
    /*
        我们在infer需要做的事情
        1. 读取model => 创建runtime, engine, context
        2. 把数据进行host->device传输
        3. 使用context推理
        4. 把数据进行device->host传输
    */

    /* 1. 读取model => 创建runtime, engine, context */
    string planFilePath = "models/sample_cpp.engine";
    if (!fileExists(planFilePath)) {
        return false;
    }

    vector<unsigned char> modelData;
    modelData = loadFile(planFilePath);
    
    Logger logger;
    auto runtime     = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine      = make_unique<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    auto context     = make_unique<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    auto input_dims   = context->getBindingDimensions(0);
    auto output_dims  = context->getBindingDimensions(1);

    cout << "input dim shape is:  " << printDims(input_dims) << endl;
    cout << "output dim shape is: " << printDims(output_dims) << endl;

    /* 2. host->device的数据传递 */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* host memory上的数据*/
    float input_host[] = {0.0193, 0.2616, 0.7713, 0.3785, 0.9980, 0.9008, 0.4766, 0.1663, 0.8045, 0.6552};
    float output_host[5];

    /* device memory上的数据*/
    float* input_device = nullptr;
    float* weight_device = nullptr;
    float* output_device = nullptr;

    int input_size = 10;
    int output_size = 5;

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
    return true;
}
