#ifndef __WORKER_HPP__
#define __WORKER_HPP__

#include <memory>
#include <vector>
#include "trt_model.hpp"
#include "logger.hpp"

/*
    我们希望做的，是使用worker作为接口进行推理在main中，我们只需要:

    worker(...);                                         //根据模型的种类(分类、检测、分割)来初始化一个模型
   
    worker->classifier(...);                             //资源获取即初始化，在这里创建engine，并且建立推理上下文。如果已经有了engine的话就直接load这个engine，并且建立推理上下文
    worker->classifier.load_image(...);                  //在这里我们读取图片，并分配pinned memory，分配device memory
    score = worker->classifier.infer_classifier(...);    //在这里我们进行预处理，推理，后处理的部分

    worker->detector(...);                               //资源获取即初始化，在这里创建engine，并且建立推理上下文。如果已经有了engine的话就直接load这个engine，并且建立推理上下文
    worker->detector.load_image(...);                    //在这里我们读取图片，并分配pinned memory，分配device memory
    bboxes = worker->detector.infer_detector(...);       //在这里我们进行预处理，推理，后处理的部分
    
    worker->segmenter(...);                              //资源获取即初始化，在这里创建engine，并且建立推理上下文。如果已经有了engine的话就直接load这个engine，并且建立推理上下文
    worker->segmenter.load_image(...);                   //在这里我们读取图片，并分配pinned memory，分配device memory
    mask = worker->segmenter.infer_segmentor(...);       //在这里我们进行预处理，推理，后处理的部分
    
    worker->drawBBox(...);                               //worker负责将bbox的信息绘制在原图上
    worker->drawMask(...);                               //worker负责将mask的信息融合在原图上
    worker->drawScore(...);                              //worker负责将score的信息绘制在原图上

    这个案例下的worker的内容是很空的。因为目前只是一个单独的classification的任务。整体上的结构很simple
    但这么设计的目的是为了今后的扩展，比如说针对视频流的异步处理，多线程的处理，multi-stage model的处理，multi-task model的处理等等

    因为比如说会出现这种情况：
    // 1st stage detection
    worker->detector(...);
    worker->detector.load_image(...);
    bboxes = worker->detector.infer_detector(...);

    // 2nd stage classification
    worker->classifier(...);
    worker->classifier.load_from_bbox(...);
    score = worker->classifier.infer_classifier(...);

*/

    
class Worker {
public:
    Worker(std::string onnxPath, Model::task_type type, Logger::Level level, Model::Params params);
    void trt_infer(std::string imagePath);
public:
    std::shared_ptr<Model> m_classifier;
    std::shared_ptr<Logger> m_logger;
    std::shared_ptr<Model::Params> m_params;
    std::vector<float> m_scores;
};


#endif //__WORKER_HPP__
