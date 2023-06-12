#ifndef __TRT_MODEL_HPP__
#define __TRT_MODEL_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"

/*
    // 我们希望根据onnx目录进行寻找engine文件。如果有就直接load，如果没有就直接build。并且根据不同的模型类型指定不同的build方式
    classifier = model("resnet18.onnx", Model::CLASSIFICATION);
    detector   = model("yolov8.onnx", Model::DETECTION);
    segmenter  = model("unet.onnx", Model::SEGMENTATION);

    // 三种不同类型的model在获取image的时候操作是一样的，都是分配pinned memory，分配device memory这些
    classifier.load_image(...);
    detector.load_image(...);
    segmenter.load_image(...);

    // 这里需要注意的是不同的task的后处理是不一样的。但仍然是有可以复用的部分，比如说enqueue这些
    claffier.infer_classifier(...);
    detector.infer_detector(...);
    segmenter.infer_segmentor(...);
    
    // 我们总结一下Model可以复用的部分
    1. model初始化的build engine和load engine部分
    2. model读取图片并分配内存的部分
    3. model进行推理是创建binding，enqueue这些部分

    //其余的地方需要根据不同的task进行不同的优化设计
*/

enum task_type {
    CLASSIFICATION,
    DETECTION,
    SEGMENTATION,
};

class Model {
public:
    Model(std::string onnx_path, task_type type);
    void load_image(std::string image_path);
    void infer_classifier();
    ~Model();

public:
    bool build_engine();
    bool load_engine();
    bool preprocess_image();
    bool postprocess_classifier();


private:
    
    std::string m_imagePath;
    std::string m_onnxPath;
    std::string m_enginePath;
    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    int m_workspaceSize;
};

#endif //__TRT_MODEL_HPP__
