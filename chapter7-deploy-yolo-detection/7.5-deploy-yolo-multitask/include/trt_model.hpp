#ifndef __TRT_MODEL_HPP__
#define __TRT_MODEL_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_timer.hpp"
#include "trt_logger.hpp"
#include "trt_preprocess.hpp"

#define WORKSPACESIZE 1<<28

namespace model{

enum task_type {
    CLASSIFICATION,
    DETECTION,
    SEGMENTATION,
    MULTITASK
};

enum device {
    CPU,
    GPU
};

enum precision {
    FP32,
    FP16,
    INT8
};

// 我们希望每一个model都有一个自己的image_info
struct image_info {
    int h;
    int w;
    int c;
    image_info(int height, int width, int channel) : h(height), w(width), c(channel) {}
};

// 对Params设定一些默认值, 这些Params是在build model中所需要的
struct Params {
    device               dev      = GPU;
    int                  num_cls  = 1000;
    preprocess::tactics  tac      = preprocess::tactics::GPU_BILINEAR;
    image_info           img      = {224, 224, 3};
    task_type            task     = CLASSIFICATION;
    int                  ws_size  = WORKSPACESIZE;
    precision            prec     = FP32;
};


/* 构建一个针对trt的shared pointer. 所有的trt指针的释放都是通过ptr->destroy完成*/
template<typename T>
void destroy_trt_ptr(T* ptr){
    if (ptr) {
        std::string type_name = typeid(T).name();
        LOGD("Destroy %s", type_name.c_str());
        ptr->destroy(); 
    };
}

class Model {

public:
    Model(std::string onnx_path, logger::Level level, Params params); 
    virtual ~Model() {};
    void load_image(std::string image_path);
    void init_model(); //初始化模型，包括build推理引擎, 分配内存，创建context, 设置bindings
    void inference();  //推理部分，preprocess-enqueue-postprocess

public:
    bool build_engine();
    bool load_engine();
    void save_plan(nvinfer1::IHostMemory& plan);
    void print_network(nvinfer1::INetworkDefinition &network, bool optimized);
    std::string getPrec(model::precision prec);

    // 这里的dnn推理部分，只要设定好了m_bindings的话，不同的task的infer_dnn的实现都是一样的
    bool enqueue_bindings();

    // 以下都是子类自己实现的内容, 通过定义一系列虚函数来实现
    // setup负责分配host/device的memory, bindings, 以及创建推理所需要的上下文。
    // 由于不同task的input/output的tensor不一样，所以这里的setup需要在子类实现
    virtual void setup(void const* data, std::size_t size) = 0;

    // 为了能够同一个model多次推理，上一次推理结束后需要根据task进行reset
    virtual void reset_task() = 0;

    // 不同的task的前处理/后处理是不一样的，所以具体的实现放在子类
    virtual bool preprocess_cpu()      = 0;
    virtual bool preprocess_gpu()      = 0;
    virtual bool postprocess_cpu()     = 0;
    virtual bool postprocess_gpu()     = 0;

public:
    std::string m_imagePath;
    std::string m_outputPath;
    std::string m_onnxPath;
    std::string m_enginePath;

    cv::Mat m_inputImage;
    Params* m_params;

    int    m_workspaceSize;
    float* m_bindings[3];
    float* m_inputMemory[2];
    float* m_outputMemory[2];

    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;
    cudaStream_t   m_stream;

    std::shared_ptr<logger::Logger>               m_logger;
    std::shared_ptr<timer::Timer>                 m_timer;
    std::shared_ptr<nvinfer1::IRuntime>           m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine>        m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext>  m_context;
    std::shared_ptr<nvinfer1::INetworkDefinition> m_network;

};

}; // namespace model

#endif //__TRT_MODEL_HPP__
