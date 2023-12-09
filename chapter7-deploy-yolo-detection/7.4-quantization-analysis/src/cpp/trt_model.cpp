#include "trt_model.hpp"
#include "utils.hpp" 
#include "trt_logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "trt_calibrator.hpp"
#include <string>

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

namespace model{

Model::Model(string onnx_path, logger::Level level, Params params) {
    m_onnxPath      = onnx_path;
    m_workspaceSize = WORKSPACESIZE;
    m_logger        = make_shared<logger::Logger>(level);
    m_timer         = make_shared<timer::Timer>();
    m_params        = new Params(params);
    m_enginePath    = changePath(onnx_path, "../engine", ".engine", getPrec(params.prec));
}

void Model::load_image(string image_path) {
    if (!fileExists(image_path)){
        LOGE("%s not found", image_path.c_str());
    } else {
        m_imagePath = image_path;
        LOG("*********************INFERENCE INFORMATION***********************");
        LOG("\tModel:      %s", getFileName(m_onnxPath).c_str());
        LOG("\tImage:      %s", getFileName(m_imagePath).c_str());
        LOG("\tPrecision:  %s", getPrec(m_params->prec).c_str());
    }
}

void Model::init_model() {
    /* 一个model的engine, context这些一旦创建好了，当多次调用这个模型的时候就没必要每次都初始化了*/
    if (m_context == nullptr){
        if (!fileExists(m_enginePath)){
            LOG("%s not found. Building trt engine...", m_enginePath.c_str());
            build_engine();
        } else {
            LOG("%s has been generated! loading trt engine...", m_enginePath.c_str());
            load_engine();
        }
    }else{
        m_timer->init();
        reset_task();
    }
}

bool Model::build_engine() {
    // 我们也希望在build一个engine的时候就把一系列初始化全部做完，其中包括
    //  1. build一个engine
    //  2. 创建一个context
    //  3. 创建推理所用的stream
    //  4. 创建推理所需要的device空间
    // 这样，我们就可以在build结束以后，就可以直接推理了。这样的写法会比较干净
    auto builder       = shared_ptr<IBuilder>(createInferBuilder(*m_logger), destroy_trt_ptr<IBuilder>);
    auto network       = shared_ptr<INetworkDefinition>(builder->createNetworkV2(1), destroy_trt_ptr<INetworkDefinition>);
    auto config        = shared_ptr<IBuilderConfig>(builder->createBuilderConfig(), destroy_trt_ptr<IBuilderConfig>);
    auto parser        = shared_ptr<IParser>(createParser(*network, *m_logger), destroy_trt_ptr<IParser>);

    config->setMaxWorkspaceSize(m_workspaceSize);
    config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED); //这里也可以设置为kDETAIL;

    if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){
        return false;
    }

    if (builder->platformHasFastFp16() && m_params->prec == model::FP16) {
        config->setFlag(BuilderFlag::kFP16);
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    } else if (builder->platformHasFastInt8() && m_params->prec == model::INT8) {
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    shared_ptr<Int8EntropyCalibrator> calibrator(new Int8EntropyCalibrator(
        64, 
        "calibration/calibration_list_coco.txt", 
        "calibration/calibration_table.txt",
        3 * 640 * 640, 640, 640));
    config->setInt8Calibrator(calibrator.get());

    auto engine        = shared_ptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config), destroy_trt_ptr<ICudaEngine>);
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);

    // 保存序列化后的engine
    save_plan(*plan);

    // 根据runtime初始化engine, context, 以及memory
    setup(plan->data(), plan->size());

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
    
    // 根据runtime初始化engine, context, 以及memory
    setup(modelData.data(), modelData.size());

    return true;
}

void Model::save_plan(IHostMemory& plan) {
    auto f = fopen(m_enginePath.c_str(), "wb");
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
}

/* 
    可以根据情况选择是否在CPU上跑pre/postprocess
    对于一些edge设备，为了最大化GPU利用效率，我们可以考虑让CPU做一些pre/postprocess，让其执行与GPU重叠
*/
void Model::inference() {
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
    m_timer->start_gpu();
    if (!m_context->enqueueV2((void**)m_bindings, m_stream, nullptr)){
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    m_timer->stop_gpu("trt-inference(GPU)");
    return true;
}

void Model::print_network(INetworkDefinition &network, bool optimized) {

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
        auto inspector = shared_ptr<IEngineInspector>(m_engine->createEngineInspector());
        for (int i = 0; i < layerCount; i++) {
            LOGV("layer_info: %s", inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }
}

string Model::getPrec(model::precision prec) {
    switch(prec) {
        case model::precision::FP16:   return "fp16";
        case model::precision::INT8:   return "int8";
        default:                       return "fp32";
    }
}

} // namespace model
