#include <memory>
#include <iostream>
#include <string>
#include <type_traits>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "utils.hpp"
#include "cuda_runtime.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "model.hpp"
#include "imagenet_labels.hpp"


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
        return true;
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


bool Model::infer(string imagePath){
    string planFilePath = this->mEnginePath;
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

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));


    int input_width    = 224;
    int input_height   = 224;
    int input_channel  = 3;
    int num_classes    = 1000;

    int input_size     = input_channel * input_width * input_height * sizeof(float);
    int output_size    = num_classes * sizeof(float);

    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};


    /*Preprocess -- allocate host and device memory*/
    float* input_host    = nullptr;
    float* input_device  = nullptr;
    float* output_host   = nullptr;
    float* output_device = nullptr;
    CUDACHECK(cudaMalloc(&input_device, input_size));
    CUDACHECK(cudaMalloc(&output_device, output_size));
    CUDACHECK(cudaMallocHost(&input_host, input_size));
    CUDACHECK(cudaMallocHost(&output_host, output_size));


    /*Preprocess -- read image*/
    cv::Mat input_image;
    input_image = cv::imread(imagePath);
    if (input_image.data == nullptr) {
        cout << "file not founded! " << endl;
        return false;
    }

    /*Preprocess -- resize*/
    cv::resize(input_image, input_image, cv::Size(input_width, input_height));

    /*Preprocess -- mean, std, rgb2bgr*/
    int image_area = input_width * input_height;
    unsigned char* pimage = input_image.data;
    float* phost_b = input_host + image_area * 0;
    float* phost_g = input_host + image_area * 1;
    float* phost_r = input_host + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }

    /*Preprocess -- move data from host to device*/
    CUDACHECK(cudaMemcpyAsync(input_device, input_host, input_size, cudaMemcpyKind::cudaMemcpyHostToDevice, stream));

    /*DNN -- infer*/
    float* bindings[] = {input_device, output_device};
    if (!context->enqueueV2((void**)bindings, stream, nullptr)){
        cout << "Error happens during DNN inference part" << endl;
        return false;
    }

    /*Postprocess -- move data from device to host*/
    CUDACHECK(cudaMemcpyAsync(output_host, output_device, output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    /*Postprocess -- argmax*/
    ImageNetLabels labels;
    int pos = max_element(output_host, output_host + num_classes) - output_host;
    float confidence = output_host[pos] * 100;
    cout << "Inference result: " << labels.imagenet_labelstring(pos) << ", Confidence is: " << confidence << "%" << endl;
    
    return true;
}
