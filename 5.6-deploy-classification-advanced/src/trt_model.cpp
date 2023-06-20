#include "trt_model.hpp"
#include "utils.hpp" 
#include "logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <string>

using namespace std;

Model::Model(string onnx_path, Logger::Level level, Params params) {
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

bool Model::inference() {
    if (m_params->dev == CPU) {
        preprocess_cpu();
    } else {
        preprocess_gpu();
    }

    enqueue_bindings();

    if (m_params->dev == CPU) {
        postprocess_cpu();
    } else {
        postprocess_gpu();
    }
}


bool Model::enqueue_bindings() {
    if (!m_context->enqueueV2((void**)m_bindings, m_stream, nullptr)){
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
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

