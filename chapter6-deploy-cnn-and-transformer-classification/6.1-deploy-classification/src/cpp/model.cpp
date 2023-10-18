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
#include "timer.hpp"
#include "imagenet_labels.hpp"


using namespace std;

class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char* msg) noexcept override{
        string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = RED    "[fatal]" CLEAR;
            case Severity::kERROR:          str = RED    "[error]" CLEAR;
            case Severity::kWARNING:        str = BLUE   "[warn]"  CLEAR;
            case Severity::kINFO:           str = YELLOW "[info]"  CLEAR;
            case Severity::kVERBOSE:        str = PURPLE "[verb]"  CLEAR;
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

Model::Model(string path, precision prec){
    if (getFileType(path) == ".onnx")
        mOnnxPath = path;
    else 
        LOGE("ERROR: %s, wrong weight or model type selected. Program terminated", getFileType(path).c_str());

    if (prec == precision::FP16) {
        mPrecision = nvinfer1::DataType::kHALF;
    } else if (prec == precision::INT8) {
        mPrecision = nvinfer1::DataType::kINT8;
    } else {
        mPrecision = nvinfer1::DataType::kFLOAT;
    }

    mEnginePath = getEnginePath(path, prec);
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
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
        LOGE("ERROR: failed to %s", mOnnxPath.c_str());
        return false;
    }

    if (builder->platformHasFastFp16() && mPrecision == nvinfer1::DataType::kHALF) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    } else if (builder->platformHasFastInt8() && mPrecision == nvinfer1::DataType::kINT8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    auto f = fopen(mEnginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    // 如果想要观察模型优化前后的架构变化，可以取消注释
    mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    mInputDims         = network->getInput(0)->getDimensions();
    mOutputDims        = network->getOutput(0)->getDimensions();

    int inputCount = network->getNbInputs();
    int outputCount = network->getNbOutputs();
    string layer_info;

    LOGV("Before TensorRT optimization");
    print_network(*network, false);
    LOGV("");
    LOGV("After TensorRT optimization");
    print_network(*network, true);

    LOGV("Finished building engine");

    return true;
};


bool Model::infer(string imagePath){
    if (!fileExists(mEnginePath)) {
        LOGE("ERROR: %s not found", mEnginePath.c_str());
        return false;
    }

    vector<unsigned char> modelData;
    modelData = loadFile(mEnginePath);
    
    Timer timer;
    Logger logger;
    auto runtime      = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine       = make_unique<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    auto context      = make_unique<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    auto input_dims   = context->getBindingDimensions(0);
    auto output_dims  = context->getBindingDimensions(1);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int input_width    = input_dims.d[3];
    int input_height   = input_dims.d[2];
    int input_channel  = input_dims.d[1];
    int num_classes    = output_dims.d[1];
    int input_size     = input_channel * input_width * input_height * sizeof(float);
    int output_size    = num_classes * sizeof(float);

    /* 
        为了让trt推理和pytorch的推理结果一致，我们需要对其pytorch所用的mean和std
        这里面opencv读取的图片是BGR格式，所以mean和std也按照BGR的顺序存储
        可以参考pytorch官方提供的前处理方案: https://pytorch.org/hub/pytorch_vision_resnet/
    */
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    /*Preprocess -- 分配host和device的内存空间*/
    float* input_host    = nullptr;
    float* input_device  = nullptr;
    float* output_host   = nullptr;
    float* output_device = nullptr;
    CUDA_CHECK(cudaMalloc(&input_device, input_size));
    CUDA_CHECK(cudaMalloc(&output_device, output_size));
    CUDA_CHECK(cudaMallocHost(&input_host, input_size));
    CUDA_CHECK(cudaMallocHost(&output_host, output_size));

    /*Preprocess -- 测速*/
    timer.start_cpu();

    /*Preprocess -- 读取数据*/
    cv::Mat input_image;
    input_image = cv::imread(imagePath);
    if (input_image.data == nullptr) {
        LOGE("file not founded! Program terminated");
        return false;
    } else {
        LOG("Model:      %s", getFileName(mOnnxPath).c_str());
        LOG("Precision:  %s", getPrecision(mPrecision).c_str());
        LOG("Image:      %s", getFileName(imagePath).c_str());
    }

    /*Preprocess -- resize(默认是bilinear interpolation)*/
    cv::resize(input_image, input_image, cv::Size(input_width, input_height));

    /*Preprocess -- host端进行normalization + BGR2RGB + hwc2cwh)*/
    int index;
    int offset_ch0 = input_width * input_height * 0;
    int offset_ch1 = input_width * input_height * 1;
    int offset_ch2 = input_width * input_height * 2;
    for (int i = 0; i < input_height; i++) {
        for (int j = 0; j < input_width; j++) {
            index = i * input_width * input_channel + j * input_channel;
            input_host[offset_ch2++] = (input_image.data[index + 0] / 255.0f - mean[0]) / std[0];
            input_host[offset_ch1++] = (input_image.data[index + 1] / 255.0f - mean[1]) / std[1];
            input_host[offset_ch0++] = (input_image.data[index + 2] / 255.0f - mean[2]) / std[2];
        }
    }

    /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(input_device, input_host, input_size, cudaMemcpyKind::cudaMemcpyHostToDevice, stream));

    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("preprocess(resize + norm + bgr2rgb + hwc2chw + H2D)");

    /*Inference -- 测速*/
    timer.start_cpu();

    /*Inference -- device端进行推理*/
    float* bindings[] = {input_device, output_device};
    if (!context->enqueueV2((void**)bindings, stream, nullptr)){
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("inference(enqueuev2)");

    /*Postprocess -- 测速*/
    timer.start_cpu();

    /*Postprocess -- 将device上的数据移动到host上*/
    CUDA_CHECK(cudaMemcpyAsync(output_host, output_device, output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*Postprocess -- 寻找label*/
    ImageNetLabels labels;
    int pos = max_element(output_host, output_host + num_classes) - output_host;
    float confidence = output_host[pos] * 100;
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("postprocess(D2H + get label)");

    LOG("Inference result: %s, Confidence is %.3f%%\n", labels.imagenet_labelstring(pos).c_str(), confidence);   

    CUDA_CHECK(cudaFree(output_device));
    CUDA_CHECK(cudaFree(input_device));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return true;
}

void Model::print_network(nvinfer1::INetworkDefinition &network, bool optimized) {

    int inputCount = network.getNbInputs();
    int outputCount = network.getNbOutputs();
    string layer_info;

    for (int i = 0; i < inputCount; i++) {
        auto input = network.getInput(i);
        LOGV("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
    }

    for (int i = 0; i < outputCount; i++) {
        auto output = network.getOutput(i);
        LOGV("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
    }
    
    int layerCount = optimized ? mEngine->getNbLayers() : network.getNbLayers();
    LOGV("network has %d layers", layerCount);

    if (!optimized) {
        for (int i = 0; i < layerCount; i++) {
            char layer_info[1000];
            auto layer   = network.getLayer(i);
            auto input   = layer->getInput(0);
            int n = 0;
            if (input == nullptr){
                continue;
            }
            auto output  = layer->getOutput(0);

            LOGV("layer_info: %-40s:%-25s->%-25s[%s]", 
                layer->getName(),
                printTensorShape(input).c_str(),
                printTensorShape(output).c_str(),
                getPrecision(layer->getPrecision()).c_str());
        }

    } else {
        auto inspector = make_unique<nvinfer1::IEngineInspector>(mEngine->createEngineInspector());
        for (int i = 0; i < layerCount; i++) {
            string info = inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kONELINE);
            info = info.substr(0, info.size() - 1);
            LOGV("layer_info: %s", info.c_str());
        }
    }
}
