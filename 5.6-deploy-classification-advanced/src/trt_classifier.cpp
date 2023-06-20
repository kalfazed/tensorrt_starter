#include "opencv2/imgproc.hpp"
#include "trt_model.hpp"
#include "utils.hpp" #include "logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "imagenet_labels.hpp"
#include "trt_classifier.hpp"
#include "preprocess.hpp"

using namespace std;

namespace classifier {

void Classifier::setup(nvinfer1::IRuntime& runtime, void const* data, size_t size) {
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

bool Classifier::preprocess_gpu() {
    /*Preprocess -- 获取mean, std*/
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    /*Preprocess -- 读取数据*/
    cv::Mat input_image;
    input_image = cv::imread(m_imagePath);
    if (input_image.data == nullptr) {
        LOGE("file not founded! Program terminated"); return false;
    }
    
    /*Preprocess -- 使用GPU进行双线性插值, 并将结果返回到m_inputMemory中*/
    preprocess_resize_gpu(input_image, m_inputMemory[1], m_params->heigth, m_params->width, mean, std, 0);

    return true;
}

bool Classifier::preprocess_cpu() {
    /*Preprocess -- 获取mean, std*/
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    /*Preprocess -- 读取数据*/
    cv::Mat input_image;
    input_image = cv::imread(m_imagePath);
    if (input_image.data == nullptr) {
        LOGE("file not founded! Program terminated"); return false;
    }

    uint8_t* pp = input_image.data;

    /*Preprocess -- resize(默认是bilinear interpolation)*/
    cv::resize(input_image, input_image, 
               cv::Size(m_params->width, m_params->heigth), 0, 0, cv::INTER_LINEAR);

    /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
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
    LOG("input value in (0,0) without gpu is %f, %f, %f", 
        m_inputMemory[0][image_area * 0], 
        m_inputMemory[0][image_area * 1], 
        m_inputMemory[0][image_area * 2]);


    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], input_size, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    return true;
}

bool Classifier::postprocess_cpu() {
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


bool Classifier::postprocess_gpu() {
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

shared_ptr<Classifier> make_classifier(
    std::string onnx_path, Logger::Level level, Model::Params params)
{
    auto classifiler_inst = make_shared<Classifier>(onnx_path, level, params);
    return classifiler_inst;
}

}; //namespace Classifier
