#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#include "utils.hpp"
#include "timer.hpp"
#include "preprocess.hpp"

using namespace std;

int main(){
    Timer timer;

    string file_path     = "data/deer.png";
    string output_prefix = "results/";
    string output_path   = "";

    cv::Mat input = cv::imread(file_path);
    int tar_h = 500;
    int tar_w = 250;
    int tactis;

    cv::Mat resizedInput_cpu;
    cv::Mat resizedInput_gpu;
    
    /* 
     * bilinear interpolation resize的CPU/GPU速度比较
     * 由于CPU端做完预处理之后，进入如果有DNN也需要将数据传送到device上，
     * 所以这里为了让测速公平，仅对下面的部分进行测速:
     *
     * - host端
     *     cv::resize的bilinear interpolation
     *     normalization进行归一化处理
     *     BGR2RGB来实现通道调换
     *
     * - device端
     *     bilinear interpolation + normalization + BGR2RGB的自定义核函数
     *
     * 由于这个章节仅是初步CUDA学习，大家在自己构建推理模型的时候可以将这些地方进行封装来写的好看点，
     * 在这里代码我们更关注实现的逻辑部分
     *
     * tatics 列表
     * 0: 最近邻差值缩放 + 全图填充
     * 1: 双线性差值缩放 + 全图填充
     * 2: 双线性差值缩放 + 填充(letter box)
     * 3: 双线性差值缩放 + 填充(letter box) + 平移居中
     * */
    
    resizedInput_cpu = preprocess_cpu(input, tar_h, tar_w, timer, tactis);
    output_path = output_prefix + getPrefix(file_path) + "_resized_bilinear_cpu.png";
    cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, resizedInput_cpu);

    tactis = 0;
    resizedInput_gpu = preprocess_gpu(input, tar_h, tar_w, timer, tactis);
    output_path = output_prefix + getPrefix(file_path) + "_resized_nearest_gpu.png";
    cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, resizedInput_gpu);

    tactis = 1;
    resizedInput_gpu = preprocess_gpu(input, tar_h, tar_w, timer, tactis);
    output_path = output_prefix + getPrefix(file_path) + "_resized_bilinear_gpu.png";
    cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, resizedInput_gpu);

    tactis = 2;
    resizedInput_gpu = preprocess_gpu(input, tar_h, tar_w, timer, tactis);
    output_path = output_prefix + getPrefix(file_path) + "_resized_bilinear_letterbox_gpu.png";
    cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, resizedInput_gpu);

    tactis = 3;
    resizedInput_gpu = preprocess_gpu(input, tar_h, tar_w, timer, tactis);
    output_path = output_prefix + getPrefix(file_path) + "_resized_bilinear_letterbox_center_gpu.png";
    cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, resizedInput_gpu);
    return 0;
}
