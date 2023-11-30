#ifndef __TRT_MULTITASK_HPP__
#define __TRT_MULTITASK_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_logger.hpp"
#include "trt_model.hpp"

namespace model{

namespace multitask {

enum model {
    YOLOV5,
    YOLOV8
};

struct bbox {
    float     x0, x1, y0, y1;
    float     confidence;
    bool      flg_remove;
    int       label;
    cv::Rect  rect;
    cv::Mat   boxMask;
    cv::Mat   mc;
    
    bbox() = default;
    bbox(float x0, float y0, float x1, float y1, float conf, int label, cv::Mat img) : 
        x0(x0), y0(y0), x1(x1), y1(y1), 
        confidence(conf), flg_remove(false), 
        label(label){
            rect = cv::Rect_<int>(x0, y0, x1 - x0, y1 - y0);

            // 如果rect越界，调整rect的大小
            if (rect.x + rect.width > img.cols)
                rect.width = img.cols - rect.x;
            if (rect.y + rect.height > img.rows)
                rect.height = img.rows - rect.y;
            
            rect.x = rect.x < 0 ? 0 : rect.x;
            rect.y = rect.y < 0 ? 0 : rect.y;
        }
};

class Multitask : public Model{

public:
    // 这个构造函数实际上调用的是父类的Model的构造函数
    Multitask(std::string onnx_path, logger::Level level, Params params) : 
        Model(onnx_path, level, params) {};

public:
    // 这里detection自己实现了一套前处理/后处理，以及内存分配的初始化
    virtual void setup(void const* data, std::size_t size) override;
    virtual void reset_task() override;
    virtual bool preprocess_cpu() override;
    virtual bool preprocess_gpu() override;
    virtual bool postprocess_cpu() override;
    virtual bool postprocess_gpu() override;

private:
    std::vector<bbox> m_bboxes;
    cv::Mat           m_masks;

    int m_inputSize; 
    int m_imgArea;
    int m_detectSize;
    int m_segmentSize;

    float* m_detectMemory[2];
    float* m_segmentMemory[2];

    nvinfer1::Dims m_detectDims;
    nvinfer1::Dims m_segmentDims;
};

// 外部调用的接口
std::shared_ptr<Multitask> make_multitask(
    std::string onnx_path, logger::Level level, Params params);

}; // namespace multitask
}; // namespace model

#endif //__TRT_MULTITASK_HPP__
