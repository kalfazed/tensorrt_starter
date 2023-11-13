#include "preprocess.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"
#include "timer.hpp"

// 根据比例进行缩放 (CPU版本)
cv::Mat preprocess_cpu(cv::Mat &src, const int &tar_h, const int &tar_w, Timer timer, int tactis) {
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
cv::Mat preprocess_gpu(
    cv::Mat &h_src, const int& tar_h, const int& tar_w, Timer timer, int tactis) 
{
    uint8_t* d_tar = nullptr;
    uint8_t* d_src = nullptr;

    cv::Mat h_tar(cv::Size(tar_w, tar_h), CV_8UC3);

    int height   = h_src.rows;
    int width    = h_src.cols;
    int chan     = 3;

    int src_size  = height * width * chan;
    int tar_size  = tar_h * tar_w * chan;

    // 分配device上的src和tar的内存
    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMalloc(&d_tar, tar_size));

    // 将数据拷贝到device上
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyHostToDevice));

    timer.start_gpu();

    // device上处理resize, BGR2RGB的核函数
    resize_bilinear_gpu(d_tar, d_src, tar_w, tar_h, width, height, tactis);

    // host和device进行同步处理
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.stop_gpu();
    switch (tactis) {
        case 0: timer.duration_gpu("Resize(nearest) in gpu takes:"); break;
        case 1: timer.duration_gpu("Resize(bilinear) in gpu takes:"); break;
        case 2: timer.duration_gpu("Resize(bilinear-letterbox) in gpu takes:"); break;
        case 3: timer.duration_gpu("Resize(bilinear-letterbox-center) in gpu takes:"); break;
        case 4: timer.duration_gpu("Resize(warpaffine-letterbox-center) in gpu takes:"); break;
        default: break;
    }

    // 将结果返回给host上
    CUDA_CHECK(cudaMemcpy(h_tar.data, d_tar, tar_size, cudaMemcpyDeviceToHost));


    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_tar));

    return h_tar;
}


