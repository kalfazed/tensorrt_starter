#include "trt_model.hpp"
#include "utils.hpp"
#include "logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "imagenet_labels.hpp"

using namespace std;

Model::Model(string onnx_path, task_type type, Logger::Level level, Params params) {
    m_onnxPath      = onnx_path;
    m_enginePath    = getEnginePath(onnx_path);
    m_workspaceSize = WORKSPACESIZE;
    m_logger        = new Logger(level);
    m_params        = new Params(params);
}

bool Model::load_image(string image_path) {
    if (!fileExists(image_path)) 
        LOGE("%s not found", image_path.c_str());
    m_imagePath = image_path;
}

void Model::init_model() {
    if (!fileExists(m_enginePath)){
        LOG("trt engine not found, building trt engine...");
        build_engine();
    } else {
        LOG("trt engine has been generated! loading trt engine...");
        load_engine();
    }
}

bool Model::build_engine() {
    // 我们也希望在build一个engine的时候就把一系列初始化全部做完，其中包括
    //  1. build一个engine
    //  2. 创建一个context
    //  3. 创建推理所用的stream
    //  4. 创建推理所需要的device空间
    // 这样，我们就可以在build结束以后，就可以直接推理了。这样的写法会比较干净
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*m_logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *m_logger));

    config->setMaxWorkspaceSize(m_workspaceSize);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED); //这里也可以设置为kDETAIL;

    if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){
        return false;
    }

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger));

    // 保存序列化后的engine
    save_plan(*plan);

    // 根据runtime初始化engine, context, 以及memory
    setup(*runtime, plan->data(), plan->size());

    // 把优化前和优化后的各个层的信息打印出来
    LOGV("Before TensorRT optimization");
    print_network(*network, false);
    LOGV("After TensorRT optimization");
    print_network(*network, true);

    return true;
}



bool Model::load_engine() {
    // 同样的，我们也希望在load一个engine的时候就把一系列初始化全部做完，其中包括
    //  1. deserialize一个engine
    //  2. 创建一个context
    //  3. 创建推理所用的stream
    //  4. 创建推理所需要的device空间
    // 这样，我们就可以在load结束以后，就可以直接推理了。这样的写法会比较干净
    
    if (!fileExists(m_enginePath)) {
        LOGE("engine does not exits! Program terminated");
        return false;
    }

    vector<unsigned char> modelData;
    modelData     = loadFile(m_enginePath);
    auto runtime  = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger));
    
    // 根据runtime初始化engine, context, 以及memory
    setup(*runtime, modelData.data(), modelData.size());

    return true;
}

void Model::save_plan(nvinfer1::IHostMemory& plan) {
    auto f = fopen(m_enginePath.c_str(), "wb");
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
}

void Model::setup(nvinfer1::IRuntime& runtime, void const* data, size_t size) {
    m_engine      = make_unique<nvinfer1::ICudaEngine>(runtime.deserializeCudaEngine(data, size));
    m_context     = make_unique<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    m_inputDims   = m_context->getBindingDimensions(0);
    m_outputDims  = m_context->getBindingDimensions(1);

    CUDA_CHECK(cudaStreamCreate(&m_stream));
    
    int input_size     = m_params->channel * m_params->width * m_params->heigth * sizeof(float);
    int output_size    = m_params->num_classes * sizeof(float);

    // 这里对host和device上的memory一起分配空间
    CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], input_size));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], input_size));
    CUDA_CHECK(cudaMalloc(&m_inputMemory[1], input_size));
    CUDA_CHECK(cudaMalloc(&m_outputMemory[1], input_size));

    // //创建m_bindings，之后再寻址就直接从这里找
    m_bindings[0] = m_inputMemory[1];
    m_bindings[1] = m_outputMemory[1];
}

bool Model::infer_classifier() {
    LOGV("input dim shape is: %s", printDims(m_inputDims).c_str());
    LOGV("output dim shape is: %s", printDims(m_outputDims).c_str());
    LOGV("input size: %d, %d, %d", m_params->heigth, m_params->width, m_params->channel);
    preprocess();
    infer_dnn();
    postprocess();
}

bool Model::preprocess() {
    /*Preprocess -- 获取mean, std*/
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    /*Preprocess -- 读取数据*/
    cv::Mat input_image;
    input_image = cv::imread(m_imagePath);
    if (input_image.data == nullptr) {
        LOGE("file not founded! Program terminated");
        return false;
    }

    /*Preprocess -- resize(默认是bilinear interpolation)*/
    cv::resize(input_image, input_image, 
               cv::Size(m_params->width, m_params->heigth));

    /*Preprocess -- host端进行normalization和BGR2RGB*/
    int image_area = m_params->width * m_params->heigth;
    unsigned char* pimage = input_image.data;
    float* phost_b = m_inputMemory[0] + image_area * 0;
    float* phost_g = m_inputMemory[0] + image_area * 1;
    float* phost_r = m_inputMemory[0] + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }

    int input_size     = m_params->channel * m_params->width * m_params->heigth * sizeof(float);

    /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], input_size, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    return true;
}

bool Model::infer_dnn() {
    if (!m_context->enqueueV2((void**)m_bindings, m_stream, nullptr)){
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    return true;
}

bool Model::postprocess() {

    /*Postprocess -- 将device上的数据移动到host上*/
    int output_size    = m_params->num_classes * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    /*Postprocess -- 手动实现argmax*/
    ImageNetLabels labels;
    int pos = max_element(m_outputMemory[0], m_outputMemory[0] + m_params->num_classes) - m_outputMemory[0];
    float confidence = m_outputMemory[0][pos] * 100;
    LOG("Inference result: %s, Confidence is %.3f%%", labels.imagenet_labelstring(pos).c_str(), confidence);   
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
    
    int layerCount = optimized ? m_engine->getNbLayers() : network.getNbLayers();
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
        auto inspector = make_unique<nvinfer1::IEngineInspector>(m_engine->createEngineInspector());
        for (int i = 0; i < layerCount; i++) {
            LOGV("layer_info: %s", inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }
}

