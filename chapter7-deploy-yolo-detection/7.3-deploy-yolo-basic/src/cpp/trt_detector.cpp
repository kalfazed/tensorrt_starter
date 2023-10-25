#include "opencv2/imgproc.hpp"
#include "trt_model.hpp"
#include "utils.hpp" 
#include "trt_logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc//imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "trt_detector.hpp"
#include "trt_preprocess.hpp"
#include "coco_labels.hpp"

using namespace std;
using namespace nvinfer1;

namespace model{

namespace detector {

void Detector::setup(void const* data, size_t size) {
   /*
     * detector setup需要做的事情
     *   创建engine, context
     *   设置bindings。这里需要注意，不同版本的yolo的输出binding可能还不一样
     *   分配memory空间。这里需要注意，不同版本的yolo的输出所需要的空间也还不一样
     */

    m_runtime     = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);
    m_engine      = shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), destroy_trt_ptr<ICudaEngine>);
    m_context     = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(), destroy_trt_ptr<IExecutionContext>);
    m_inputDims   = m_context->getBindingDimensions(0);
    m_outputDims  = m_context->getBindingDimensions(1);

    CUDA_CHECK(cudaStreamCreate(&m_stream));
    
    m_inputSize     = m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
    m_imgArea       = m_params->img.h * m_params->img.w;
    m_outputSize    = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);

    // 这里对host和device上的memory一起分配空间
    CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], m_outputSize));
    CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
    CUDA_CHECK(cudaMalloc(&m_outputMemory[1], m_outputSize));

    // 创建m_bindings，之后再寻址就直接从这里找
    m_bindings[0] = m_inputMemory[1];
    m_bindings[1] = m_outputMemory[1];
}

bool Detector::preprocess_cpu() {
    /*Preprocess -- yolo的预处理并没有mean和std，所以可以直接skip掉mean和std的计算 */

    /*Preprocess -- 读取数据*/
    m_inputImage = cv::imread(m_imagePath);
    if (m_inputImage.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    /*Preprocess -- 测速*/
    m_timer->start_cpu();

    /*Preprocess -- resize(默认是bilinear interpolation)*/
    cv::resize(m_inputImage, m_inputImage, 
               cv::Size(m_params->img.w, m_params->img.h), 0, 0, cv::INTER_LINEAR);

    /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
    int index;
    int offset_ch0 = m_imgArea * 0;
    int offset_ch1 = m_imgArea * 1;
    int offset_ch2 = m_imgArea * 2;
    for (int i = 0; i < m_inputDims.d[2]; i++) {
        for (int j = 0; j < m_inputDims.d[3]; j++) {
            index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
            m_inputMemory[0][offset_ch2++] = m_inputImage.data[index + 0] / 255.0f;
            m_inputMemory[0][offset_ch1++] = m_inputImage.data[index + 1] / 255.0f;
            m_inputMemory[0][offset_ch0++] = m_inputImage.data[index + 2] / 255.0f;
        }
    }

    /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");
    return true;
}

bool Detector::preprocess_gpu() {
    /*Preprocess -- yolo的预处理并没有mean和std，所以可以直接skip掉mean和std的计算 */

    /*Preprocess -- 读取数据*/
    m_inputImage = cv::imread(m_imagePath);
    if (m_inputImage.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated"); return false;
    }
    
    /*Preprocess -- 测速*/
    m_timer->start_gpu();

    /*Preprocess -- 使用GPU进行warpAffine, 并将结果返回到m_inputMemory中*/
    preprocess::preprocess_resize_gpu(m_inputImage, m_inputMemory[1],
                                   m_params->img.h, m_params->img.w, 
                                   preprocess::tactics::GPU_WARP_AFFINE);

    m_timer->stop_gpu();
    m_timer->duration_gpu("preprocess(GPU)");
    return true;
}


bool Detector::postprocess_cpu() {
    m_timer->start_cpu();

    /*Postprocess -- 将device上的数据移动到host上*/
    int output_size    = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    /*Postprocess -- yolov8的postprocess需要做的事情*/
    /*
     * 1. 把bbox从输出tensor拿出来，并进行decode，把获取的bbox放入到m_bboxes中
     * 2. 把decode得到的m_bboxes根据nms threshold进行NMS处理
     * 3. 把最终得到的bbox绘制到原图中
     */

    float conf_threshold = 0.25;
    float nms_threshold = 0.5;

    /*Postprocess -- decode*/
    /*
     * 我们需要做的就是将[batch, bboxes, ch]转换为vector<bbox>
     * 几个步骤:
     * 1. 从每一个bbox中对应的ch中获取cx, cy, width, height, confidence
     * 2. 对每一个bbox中对应的ch中，找到最大的class label, 可以使用std::max_element
     * 3. 将cx, cy, width, height转换为x0, y0, x1, y1
     * 4. 因为图像是经过resize了的，所以需要根据resize的scale和shift进行坐标的转换
     * 5. 将转换好的x0, y0, x1, y1，以及confidence和classness给存入到box中，并push到m_bboxes中，准备接下来的NMS处理
     */
    int bboxes_count = m_outputDims.d[1];
    int yolo_feature = m_outputDims.d[2];

    for (int i = 0; i < bboxes_count; i ++) {
        /* feature: 4 + 1 + classes */
        float* ptr = m_outputMemory[0] + i * yolo_feature;
        float objectness = ptr[4];

        if (objectness < conf_threshold) 
            continue;

        int label = max_element(ptr + 5, ptr + yolo_feature) - (ptr + 5);
        float prob = ptr[5 + label];
        float confidence = objectness * prob;

        if (confidence < conf_threshold)
            continue;
        
        float cx       = ptr[0];
        float cy       = ptr[1];
        float width    = ptr[2];
        float height   = ptr[3];

        float x0       = cx - width / 2;
        float y0       = cy - height / 2;
        float x1       = cx + width / 2;
        float y1       = cy + height / 2;

        // resize过的坐标可以通过scale和shift进行reverse，得到原本的坐标
        preprocess::affine_transformation(preprocess::affine_matrix.reverse, x0, y0, &x0, &y0);
        preprocess::affine_transformation(preprocess::affine_matrix.reverse, x1, y1, &x1, &y1);

        bbox box{x0, y0, x1, y1, confidence, label};
        LOGV("id: %d, label: %d, objectness is %lf, lt[%lf, %lf], rb[%lf, %lf]", i, label, objectness, x0, y0, x1, y1);
        m_bboxes.push_back(box);
    }
    LOGV("the count of decoded bbox is %d", m_bboxes.size());


    /*Postprocess -- NMS*/
    /* 
     * 几个步骤:
     * 1. 做一个IoU计算的lambda函数
     * 2. 将m_bboxes中的所有数据，按照confidence从高到低进行排序
     * 3. 最终希望是对于每一个class，我们都只有一个bbox，所以对同一个class的所有bboxes进行IoU比较，
     *    选取confidence最大。并与其他的同类bboxes的IoU的重叠率最大的同时IoU > IoU threshold
     */
    vector<bbox> final_bboxes;
    final_bboxes.reserve(m_bboxes.size());

    auto iou = [](bbox b1, bbox b2){
        float inter_x0 = max(b1.x0, b2.x0);
        float inter_y0 = max(b1.y0, b2.y0);
        float inter_x1 = min(b1.x1, b2.x1);
        float inter_y1 = min(b1.y1, b2.y1);

        float area_inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0);
        float area_union = 
            (b1.x1 - b1.x0) * (b1.y1 - b1.y0) + 
            (b2.x1 - b2.x0) * (b2.y1 - b2.y0) - 
            area_inter;
                           
        return area_inter / area_union;
    };

    sort(m_bboxes.begin(), m_bboxes.end(), 
         [](bbox& b1, bbox& b2){return b1.confidence > b2.confidence;});

    for (int i = 0; i < m_bboxes.size(); i ++) {
        if (m_bboxes[i].flg_remove) 
            continue;

        final_bboxes.emplace_back(m_bboxes[i]);
        
        for (int j = i; j < m_bboxes.size(); j ++) {
            if (m_bboxes[j].flg_remove)
                continue;

            if (m_bboxes[i].label == m_bboxes[j].label) {
                if (iou(m_bboxes[i], m_bboxes[j]) > nms_threshold)
                    m_bboxes[j].flg_remove = true;
            }
        }
    }
    LOGV("the count of bbox after NMS is %d", final_bboxes.size());


    /*Postprocess -- draw_bbox*/
    /*
     * 几个步骤
     * 1. 通过label获取name
     * 2. 通过label获取color
     * 3. cv::rectangle
     * 4. cv::putText
     */

    m_outputPath = getOutputPath(m_imagePath, "detect");
    CocoLabels labels;
    for (auto item: final_bboxes) {
        auto name  = labels.coco_get_label(item.label);
        auto color = labels.coco_get_color(item.label);

        auto txt   = cv::format("%s:%.2f", name.c_str(), item.confidence);

        cv::Rect   rect_box{(int)item.x0, (int)item.y0, (int)(item.x1 - item.x0), (int)(item.y1 - item.y0)};

        cv::Size   size = cv::getTextSize(txt, 0, 0.9, 1, nullptr);
        cv::Point  loc((int)item.x0, (int)item.y0 - size.height);
        cv::Rect   rect_txt{(int)item.x0, (int)item.y0 - size.height - 20, size.width, size.height + 18};

        cv::rectangle(m_inputImage, rect_box, color, 3);
        cv::rectangle(m_inputImage, rect_txt, color, -1);
        cv::putText(m_inputImage, txt, loc, 0, 0.9, cv::Scalar(255, 255, 255));

        LOG("%-10s detected. Confidence: %.2f. Cord: lt[%.2f, %.2f], rb[%.2f, %.2f]", 
            name.c_str(), item.confidence, item.x0, item.y0, item.x1, item.y1);
    }
    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");

    cv::imwrite(m_outputPath, m_inputImage);

    return true;
}


bool Detector::postprocess_gpu() {
    return postprocess_cpu();
}

shared_ptr<Detector> make_detector(
    std::string onnx_path, logger::Level level, Params params)
{
    return make_shared<Detector>(onnx_path, level, params);
}

}; // namespace detector
}; // namespace model
