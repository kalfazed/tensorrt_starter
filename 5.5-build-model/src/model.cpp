#include <cstring>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <type_traits>
#include <assert.h>

#include "model.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "utils.hpp"
#include "cuda_runtime.h"
#include "math.h"

float input_5x5[] = {
    0.7576, 0.2793, 0.4031, 0.7347, 0.0293,
    0.7999, 0.3971, 0.7544, 0.5695, 0.4388,
    0.6387, 0.5247, 0.6826, 0.3051, 0.4635,
    0.4550, 0.5725, 0.4980, 0.9371, 0.6556,
    0.3138, 0.1980, 0.4162, 0.2843, 0.3398};

float input_1x5[] = {
    0.7576, 0.2793, 0.4031, 0.7347, 0.0293};


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

Model::Model(string path){
    if (getFileType(path) == ".onnx")
        mOnnxPath = path;
    else if (getFileType(path) == ".weights")
        mWtsPath = path;
    else 
        LOGE("ERROR: %s, wrong weight or model type selected. Program terminated", getFileType(path).c_str());

    mEnginePath = getEnginePath(path);
}

// decode一个weights文件，并保存到map中
// weights的格式是:
//    count
//    [name][len][weights value in hex mode]
//    [name][len][weights value in hex mode]
//    ...
map<string, nvinfer1::Weights> Model::loadWeights(){
    ifstream f;
    if (!fileExists(mWtsPath)){ 
        LOGE("ERROR: %s not found", mWtsPath.c_str());
    }

    f.open(mWtsPath);

    int32_t size;
    map<string, nvinfer1::Weights> maps;
    f >> size;

    if (size <= 0) {
        LOGE("ERROR: no weights found in %s", mWtsPath.c_str());
    }

    while (size > 0) {
        nvinfer1::Weights weight;
        string name;
        int weight_length;

        f >> name;
        f >> std::dec >> weight_length;

        uint32_t* values = (uint32_t*)malloc(sizeof(uint32_t) * weight_length);
        for (int i = 0; i < weight_length; i ++) {
            f >> std::hex >> values[i];
        }

        weight.type = nvinfer1::DataType::kFLOAT;
        weight.count = weight_length;
        weight.values = values;

        maps[name] = weight;

        size --;
    }

    return maps;
}

bool Model::build() {
    if (mOnnxPath != "") {
        return build_from_onnx();
    } else {
        return build_from_weights();
    }
}

bool Model::build_from_weights(){
    if (fileExists(mEnginePath)){
        LOG("%s has been generated!", mEnginePath.c_str());
        return true;
    } else {
        LOG("%s not found. Building engine...", mEnginePath.c_str());
    }

    mWts = loadWeights();

    // 这里和之前的创建方式是一样的
    Logger logger;
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    // 根据不同的网络架构创建不同的TensorRT网络，这里使用几个简单的例子
    if (mWtsPath == "models/sample_linear.weights") {
        build_linear(*network, mWts);
    } else if (mWtsPath == "models/sample_conv.weights") {
        build_conv(*network, mWts);
    } else if (mWtsPath == "models/sample_permute.weights") {
        build_permute(*network, mWts);
    } else if (mWtsPath == "models/sample_reshape.weights") {
        build_reshape(*network, mWts);
    } else if (mWtsPath == "models/sample_batchNorm.weights") {
        build_batchNorm(*network, mWts);
    } else if (mWtsPath == "models/sample_cbr.weights") {
        build_cbr(*network, mWts);
    } else {
        return false;
    }

    // 接下来的事情也是一样的
    config->setMaxWorkspaceSize(1<<28);
    builder->setMaxBatchSize(1);

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    cout << "file size is " << plan->size() << endl;
    auto f = fopen(mEnginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    mInputDims         = network->getInput(0)->getDimensions();
    mOutputDims        = network->getOutput(0)->getDimensions();

    // 把优化前和优化后的各个层的信息打印出来
    LOG("Before TensorRT optimization");
    print_network(*network, false);
    LOG("");
    LOG("After TensorRT optimization");
    print_network(*network, true);

    // 最后把map给free掉
    for (auto& mem : mWts) {
        free((void*) (mem.second.values));
    }
    LOG("Finished building engine");
    return true;
}

bool Model::build_from_onnx(){
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

    // 把优化前和优化后的各个层的信息打印出来
    LOG("Before TensorRT optimization");
    print_network(*network, false);
    LOG("");
    LOG("After TensorRT optimization");
    print_network(*network, true);

    LOG("Finished building engine");
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
    if (!fileExists(mEnginePath)) {
        LOGE("ERROR: %s not found", mEnginePath.c_str());
        return false;
    }

    vector<unsigned char> modelData;
    modelData = loadFile(mEnginePath);
    
    Logger logger;
    auto runtime     = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine      = make_unique<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    auto context     = make_unique<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    auto input_dims   = context->getBindingDimensions(0);
    auto output_dims  = context->getBindingDimensions(1);

    LOG("input dim shape is:  %s", printDims(input_dims).c_str());
    LOG("output dim shape is: %s", printDims(output_dims).c_str());

    /* 2. 创建流 */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* 2. 初始化input，以及在host/device上分配空间 */
    init_data();

    /* 2. host->device的数据传递*/
    cudaMemcpyAsync(mInputDevice, mInputHost, mInputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

    /* 3. 模型推理, 最后做同步处理 */
    float* bindings[] = {mInputDevice, mOutputDevice};
    bool success = context->enqueueV2((void**)bindings, stream, nullptr);

    /* 4. device->host的数据传递 */
    cudaMemcpyAsync(mOutputHost, mOutputDevice, mOutputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    LOG("input data is:  %s", printTensor(mInputHost, mInputSize / sizeof(float), input_dims).c_str());
    LOG("output data is: %s", printTensor(mOutputHost, mOutputSize / sizeof(float), output_dims).c_str());
    LOG("finished inference");
    return true;
}

void Model::print_network(nvinfer1::INetworkDefinition &network, bool optimized) {

    int inputCount = network.getNbInputs();
    int outputCount = network.getNbOutputs();
    string layer_info;

    for (int i = 0; i < inputCount; i++) {
        auto input = network.getInput(i);
        LOG("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
    }

    for (int i = 0; i < outputCount; i++) {
        auto output = network.getOutput(i);
        LOG("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
    }
    
    int layerCount = optimized ? mEngine->getNbLayers() : network.getNbLayers();
    LOG("network has %d layers", layerCount);

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

            LOG("layer_info: %-40s:%-25s->%-25s[%s]", 
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
            LOG("layer_info: %s", info.c_str());
        }
    }
}

// 最基本的Fully Connected层的创建
void Model::build_linear(nvinfer1::INetworkDefinition& network, map<string, nvinfer1::Weights> mWts) {
    auto data          = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 1, 5});
    auto fc            = network.addFullyConnected(*data, 1, mWts["linear.weight"], {});
    fc->setName("linear1");

    fc->getOutput(0) ->setName("output0");
    network.markOutput(*fc->getOutput(0));
}

// 最基本的convolution层的创建
void Model::build_conv(nvinfer1::INetworkDefinition& network, map<string, nvinfer1::Weights> mWts) {
    auto data          = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 5, 5});
    auto conv          = network.addConvolutionNd(*data, 3, nvinfer1::DimsHW{3, 3}, mWts["conv.weight"], mWts["conv.bias"]);
    conv->setName("conv1");
    conv->setStride(nvinfer1::DimsHW(1, 1));

    conv->getOutput(0) ->setName("output0");
    network.markOutput(*conv->getOutput(0));
}

// shuffle层的创建
void Model::build_permute(nvinfer1::INetworkDefinition& network, map<string, nvinfer1::Weights> mWts) {
    auto data          = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 5, 5});
    auto conv          = network.addConvolutionNd(*data, 3, nvinfer1::DimsHW{3, 3}, mWts["conv.weight"], mWts["conv.bias"]);
    conv->setName("conv1");
    conv->setStride(nvinfer1::DimsHW(1, 1));
    
    auto permute       = network.addShuffle(*conv->getOutput(0));
    permute->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1}); // B, C, H, W -> B, H, W, C
    permute->setName("permute1");

    permute->getOutput(0)->setName("output0");
    network.markOutput(*permute->getOutput(0));
}

// shuffle层中处理多次维度上的操作
void Model::build_reshape(nvinfer1::INetworkDefinition& network, map<string, nvinfer1::Weights> mWts) {
    auto data          = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 5, 5});

    auto conv          = network.addConvolutionNd(*data, 3, nvinfer1::DimsHW{3, 3}, mWts["conv.weight"], mWts["conv.bias"]);
    conv->setName("conv1");
    conv->setStride(nvinfer1::DimsHW(1, 1));

    auto reshape       = network.addShuffle(*conv->getOutput(0));
    reshape->setReshapeDimensions(nvinfer1::Dims3{1, 3, -1});
    reshape->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});      
    reshape->setName("reshape + permute1");
    // 注意，因为reshape和transpose都是属于iShuffleLayer做的事情，所以需要指明是reshape在前，还是transpose在前
    // 通过这里我们可以看到，在TRT中连续的对tensor的维度的操作其实是可以在TRT中用一个层来处理，属于一种layer fusion优化

    reshape->getOutput(0)->setName("output0");
    network.markOutput(*reshape->getOutput(0));
}

// 自定义的IScaleLayer来实现BatchNorm的创建
void Model::build_batchNorm(nvinfer1::INetworkDefinition& network, map<string, nvinfer1::Weights> mWts) {
    auto data          = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 5, 5});
    auto conv          = network.addConvolutionNd(*data, 3, nvinfer1::DimsHW{3, 3}, mWts["conv.weight"], mWts["conv.bias"]);
    conv->setName("conv1");
    conv->setStride(nvinfer1::DimsHW(1, 1));

    // 因为TensorRT内部没有BatchNorm的实现，但是我们只要知道BatchNorm的计算原理，就可以使用IScaleLayer来创建BN的计算
    // IScaleLayer主要是用在quantization和dequantization，作为提前了解，我们试着使用IScaleLayer来搭建于一个BN的parser
    // IScaleLayer可以实现: y = (x * scale + shift) ^ pow

    float* gamma   = (float*)mWts["norm.weight"].values;
    float* beta    = (float*)mWts["norm.bias"].values;
    float* mean    = (float*)mWts["norm.running_mean"].values;
    float* var     = (float*)mWts["norm.running_var"].values;
    float  eps     = 1e-5;
    
    int    count   = mWts["norm.running_var"].count;

    float* scales  = (float*)malloc(count * sizeof(float));
    float* shifts  = (float*)malloc(count * sizeof(float));
    float* pows    = (float*)malloc(count * sizeof(float));
    
    // 这里具体参考一下batch normalization的计算公式，网上有很多
    for (int i = 0; i < count; i ++) {
        scales[i] = gamma[i] / sqrt(var[i] + eps);
        shifts[i] = beta[i] - (mean[i] * gamma[i] / sqrt(var[i] + eps));
        pows[i]   = 1.0;
    }

    // 将计算得到的这些值写入到Weight中
    auto scales_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, scales, count};
    auto shifts_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, shifts, count};
    auto pows_weights   = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, pows, count};

    // 创建IScaleLayer并将这些weights传进去，这里使用channel作为scale model
    auto scale = network.addScale(*conv->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, shifts_weights, scales_weights, pows_weights);
    scale->setName("batchNorm1");

    scale->getOutput(0) ->setName("output0");
    network.markOutput(*scale->getOutput(0));
}

// 自定义的IScaleLayer来实现BatchNorm的创建
void Model::build_cbr(nvinfer1::INetworkDefinition& network, map<string, nvinfer1::Weights> mWts) {
    auto data          = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 5, 5});
    auto conv          = network.addConvolutionNd(*data, 3, nvinfer1::DimsHW{3, 3}, mWts["conv.weight"], mWts["conv.bias"]);
    conv->setName("conv1");
    conv->setStride(nvinfer1::DimsHW(1, 1));

    // 因为TensorRT内部没有BatchNorm的实现，但是我们只要知道BatchNorm的计算原理，就可以使用IScaleLayer来创建BN的计算
    // IScaleLayer主要是用在quantization和dequantization，作为提前了解，我们试着使用IScaleLayer来搭建于一个BN的parser
    // IScaleLayer可以实现: y = (x * scale + shift) ^ pow

    float* gamma   = (float*)mWts["norm.weight"].values;
    float* beta    = (float*)mWts["norm.bias"].values;
    float* mean    = (float*)mWts["norm.running_mean"].values;
    float* var     = (float*)mWts["norm.running_var"].values;
    float  eps     = 1e-5;
    
    int    count   = mWts["norm.running_var"].count;

    float* scales  = (float*)malloc(count * sizeof(float));
    float* shifts  = (float*)malloc(count * sizeof(float));
    float* pows    = (float*)malloc(count * sizeof(float));
    
    // 这里具体参考一下batch normalization的计算公式，网上有很多
    for (int i = 0; i < count; i ++) {
        scales[i] = gamma[i] / sqrt(var[i] + eps);
        shifts[i] = beta[i] - (mean[i] * gamma[i] / sqrt(var[i] + eps));
        pows[i]   = 1.0;
    }

    // 将计算得到的这些值写入到Weight中
    auto scales_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, scales, count};
    auto shifts_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, shifts, count};
    auto pows_weights   = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, pows, count};

    // 创建IScaleLayer并将这些weights传进去，这里使用channel作为scale model
    auto bn = network.addScale(*conv->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, shifts_weights, scales_weights, pows_weights);
    bn->setName("batchNorm1");

    auto leaky = network.addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kLEAKY_RELU);
    leaky->setName("leaky1");

    leaky->getOutput(0) ->setName("output0");
    network.markOutput(*leaky->getOutput(0));
}

void Model::init_data(){
    if (mWtsPath == "models/sample_linear.weights") {
        mInputSize = 5 * sizeof(float);
        mOutputSize = 1 * sizeof(float);
    } else {
        mInputSize = 25 * sizeof(float);
        mOutputSize = 27 * sizeof(float);
    }

    cudaMallocHost(&mInputHost, mInputSize);
    cudaMallocHost(&mOutputHost, mOutputSize);

    if (mWtsPath == "models/sample_linear.weights") {
        mInputHost = input_1x5;
    } else {
        mInputHost = input_5x5;
    }

    cudaMalloc(&mInputDevice, mInputSize);
    cudaMalloc(&mOutputDevice, mOutputSize);
    
}

