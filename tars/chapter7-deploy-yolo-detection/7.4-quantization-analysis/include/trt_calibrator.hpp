#ifndef __TRT_CALIBRATOR_HPP__
#define __TRT_CALIBRATOR_HPP__

#include "NvInfer.h"
#include <string>
#include <vector>


namespace model{
/*
 * 自定义一个calibrator类
 * 我们在创建calibrator的时候需要继承nvinfer1中的calibrator类
 * TensorRT提供了五种Calibrator类
 *
 *   - nvinfer1::IInt8EntropyCalibrator2
 *   - nvinfer1::IInt8MinMaxCalibrator
 *   - nvinfer1::IInt8EntropyCalibrator
 *   - nvinfer1::IInt8LegacyCalibrator
 *   - nvinfer1::IInt8Calibrator
 * 具体有什么不同，建议读一下官方文档和回顾一下之前的学习资料
 *
 * 默认下推荐使用IInt8EntropyCalibrator2
*/

// class Int8EntropyCalibrator: public nvinfer1::IInt8EntropyCalibrator2 {
class Int8EntropyCalibrator: public nvinfer1::IInt8MinMaxCalibrator {
// class Int8EntropyCalibrator: public nvinfer1::IInt8LegacyCalibrator {
// class Int8EntropyCalibrator: public nvinfer1::IInt8MinMaxCalibrator {
// class Int8EntropyCalibrator: public nvinfer1::IInt8Calibrator {

public:
    Int8EntropyCalibrator(
        const int& batchSize,
        const std::string& calibrationSetPath,
        const std::string& calibrationTablePath,
        const int& inputSize,
        const int& inputH,
        const int& inputW);

    ~Int8EntropyCalibrator(){};

    int         getBatchSize() const noexcept override {return m_batchSize;};
    bool        getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(std::size_t &length) noexcept override;
    void        writeCalibrationCache (const void* ptr, std::size_t legth) noexcept override;

private:
    const int   m_batchSize;
    const int   m_inputH;
    const int   m_inputW;
    const int   m_inputSize;
    const int   m_inputCount;
    const std::string m_calibrationTablePath {nullptr};
    
    std::vector<std::string> m_imageList;
    std::vector<char>        m_calibrationCache;

    float* m_deviceInput{nullptr};
    bool   m_readCache{true};
    int    m_imageIndex;
    
};

}; // namespace model

#endif __TRT_CALIBRATOR_HPP__
