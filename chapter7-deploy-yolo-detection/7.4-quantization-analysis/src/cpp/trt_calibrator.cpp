#include "NvInfer.h"
#include "trt_calibrator.hpp"
#include "utils.hpp"
#include "trt_logger.hpp"
#include "trt_preprocess.hpp"

#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace nvinfer1;

namespace model{

/*
 * calibrator的构造函数
 * 我们在这里把calibration所需要的数据集准备好，需要保证数据集的数量可以被batchSize整除
 * 同时由于calibration是在device上进行的，所以需要分配空间
 */
Int8EntropyCalibrator::Int8EntropyCalibrator(
    const int&    batchSize,
    const string& calibrationDataPath,
    const string& calibrationTablePath,
    const int&    inputSize,
    const int&    inputH,
    const int&    inputW):

    m_batchSize(batchSize),
    m_inputH(inputH),
    m_inputW(inputW),
    m_inputSize(inputSize),
    m_inputCount(batchSize * inputSize),
    m_calibrationTablePath(calibrationTablePath)
{
    m_imageList = loadDataList(calibrationDataPath);
    m_imageList.resize(static_cast<int>(m_imageList.size() / m_batchSize) * m_batchSize);
    std::random_shuffle(m_imageList.begin(), m_imageList.end(), 
                        [](int i){ return rand() % i; });
    CUDA_CHECK(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));
}

/*
 * 获取做calibration的时候的一个batch的图片，之后上传到device上
 * 需要注意的是，这里面的一个batch中的每一个图片，都需要做与真正推理是一样的前处理
 * 这里面我们选择在GPU上进行前处理，所以处理万
 */
bool Int8EntropyCalibrator::getBatch(
    void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (m_imageIndex + m_batchSize >= m_imageList.size() + 1)
        return false;

    LOG("%3d/%3d (%3dx%3d): %s", 
        m_imageIndex + 1, m_imageList.size(), m_inputH, m_inputW, m_imageList.at(m_imageIndex).c_str());
    
    /*
     * 对一个batch里的所有图像进行预处理
     * 这里可有以及个扩展的点
     *  1. 可以把这个部分做成函数，以函数指针的方式传给calibrator。因为不同的task会有不同的预处理
     *  2. 可以实现一个bacthed preprocess
     * 这里留给当作今后的TODO
     */
    cv::Mat input_image;
    for (int i = 0; i < m_batchSize; i ++){
        input_image = cv::imread(m_imageList.at(m_imageIndex++));
        preprocess::preprocess_resize_gpu(
            input_image, 
            m_deviceInput + i * m_inputSize,
            m_inputH, m_inputW, 
            preprocess::tactics::GPU_BILINEAR_CENTER);
    }

    bindings[0] = m_deviceInput;

    return true;
}
    
/* 
 * 读取calibration table的信息来创建INT8的推理引擎, 
 * 将calibration table的信息存储到calibration cache，这样可以防止每次创建int推理引擎的时候都需要跑一次calibration
 * 如果没有calibration table的话就会直接跳过这一步，之后调用writeCalibrationCache来创建calibration table
 */
const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept
{
    void* output;
    m_calibrationCache.clear();

    ifstream input(m_calibrationTablePath, ios::binary);
    input >> noskipws;
    if (m_readCache && input.good())
        copy(istream_iterator<char>(input), istream_iterator<char>(), back_inserter(m_calibrationCache));

    length = m_calibrationCache.size();
    if (length){
        LOG("Using cached calibration table to build INT8 trt engine...");
        output = &m_calibrationCache[0];
    }else{
        LOG("Creating new calibration table to build INT8 trt engine...");
        output = nullptr;
    }
    return output;
}

/* 
 * 将calibration cache的信息写入到calibration table中
*/
void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    ofstream output(m_calibrationTablePath, ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}

} // namespace model
