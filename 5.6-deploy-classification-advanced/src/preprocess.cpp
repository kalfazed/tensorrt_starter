#include "preprocess.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"
#include "timer.hpp"

// 根据比例进行缩放 (CPU版本)
cv::Mat preprocess_resize_cpu(cv::Mat &src, const int &tar_h, const int &tar_w, Timer timer, int tactis) {
    cv::Mat tar;

    int height  = src.rows;
    int width   = src.cols;
    float dim   = std::max(height, width);
    int resizeH = ((height / dim) * tar_h);
    int resizeW = ((width / dim) * tar_w);

    int xOffSet = (tar_w - resizeW) / 2;
    int yOffSet = (tar_h - resizeH) / 2;

    resizeW    = tar_w;
    resizeH    = tar_h;

    timer.start_cpu();

    /*BGR2RGB*/
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);

    /*Resize*/
    cv::resize(src, tar, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);

    

    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Resize(bilinear) in cpu takes:");

    return tar;
}

// 根据比例进行缩放 (GPU版本)
void preprocess_resize_gpu(
    cv::Mat &h_src, float* d_tar, 
    const int& tar_h, const int& tar_w, 
    float* h_mean, float* h_std, int tactis) 
{
    float*   d_mean = nullptr;
    float*   d_std  = nullptr;
    uint8_t* d_src  = nullptr;

    // cv::Mat h_tar(cv::Size(tar_w, tar_h), CV_8UC3);

    int height   = h_src.rows;
    int width    = h_src.cols;
    int chan     = 3;

    int src_size  = height * width * chan * sizeof(uint8_t);
    int norm_size = 3 * sizeof(float);

    // 分配device上的src和mean, std的内存
    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMalloc(&d_mean, norm_size));
    CUDA_CHECK(cudaMalloc(&d_std, norm_size));

    // 将数据拷贝到device上
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean, norm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_std, h_std, norm_size, cudaMemcpyHostToDevice));

    // device上处理resize, BGR2RGB的核函数
    resize_bilinear_gpu(d_tar, d_src, tar_w, tar_h, width, height, d_mean, d_std, tactis);

    // host和device进行同步处理
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_std));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_src));

    // 因为接下来会继续在gpu上进行处理，所以这里不用把结果返回到host
}
