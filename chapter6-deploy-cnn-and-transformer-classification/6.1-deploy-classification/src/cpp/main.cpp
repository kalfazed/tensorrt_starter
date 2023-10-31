#include <iostream>
#include <memory>

#include "model.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    Model model("models/onnx/swin-tiny-opset17", Model::precision::FP32);
    // Model model("models/onnx/swin-tiny-opset12.onnx", Model::precision::FP32);

    if(!model.build()){
        LOGE("fail in building model");
        return 0;
    }

    if(!model.infer("data/fox.png")){
        LOGE("fail in infering model");
        return 0;
    }
    if(!model.infer("data/cat.png")){
        LOGE("fail in infering model");
        return 0;
    }

    if(!model.infer("data/eagle.png")){
        LOGE("fail in infering model");
        return 0;
    }

    if(!model.infer("data/gazelle.png")){
        LOGE("fail in infering model");
        return 0;
    }

    return 0;
}

/*
    这是一个根据第五章节的代码的改编，构建出来的一个classification的推理框架。
    可以实现一系列 preprocess + enqueue + postprocess的推理。这里大家可以简单的参考一下实现的方法,
    但是建议大家如果自己从零构建推理框架的话，不要这么写, 主要因为其实这个框架有非常多的缺陷导致有很多潜在性的问题。这里罗列几点

    1. 代码看起来比较乱，主要有几个原因
        - 整体上没有使用C++设计模式, 导致代码复用不好、可读性差、灵活性也比较低，对以后的代码的扩展不是很友好
            - 比如说，如果想让这个框架支持detection或者segmentation的话，需要怎么办?
            - 再比如说，如果想要想让框架做成multi-stage, multi-task的模型的话，需要怎么办?
            - 再比如说，如果想要这个框架既支持image输入，也支持3D point cloud输入，需要怎么办?

        - 封装没有做好, 导致有一些没有必要的信息和操作暴露在外面，让代码的可读性差
            - 比如说，我们是否需要从main函数中考虑如果build/infer失败应该怎么办?
            - 再比如说，我们在创建一个engine的时候，是否可以只从main中给提供一系列参数, 其余的信息不暴露?

        - 内存复用没有做好, 导致出现一些额外的开辟销毁内存的开销
            - 比如说，多张图片依次推理的时候，是否需要每次都cudaMalloc或者cudaMallocHost, 以及cudaStreamCreate?
            - 在比如说，我们是否可以在model的构造函数初始化的时候，就把这个model所需要的CPU和GPU内存分配好以后就不变了呢？

    2. 其次仍然还有很多功能没有写全
        - INT8量化的calibrator的实现
        - trt plugin的实现
        - CPU和GPU overlap推理的实现
        - 当前batch是1, 对于多batch,或者动态batch的时候的处理方案
        - CPU端的thread级别的并行处理
        - ...
    
    当然还有很多需要扩展的地方，但是我们可以根据目前出现的一些问题，考虑一些解决方案。具体可以参考6.2-deploy-classification-advanced
*/
