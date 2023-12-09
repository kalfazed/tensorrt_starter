#include <chrono>
#include <iostream>
#include <memory>
#include "trt_timer.hpp"

#include "utils.hpp"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

namespace timer {

Timer::Timer(){
    _timeElasped = 0;
    _cStart = std::chrono::high_resolution_clock::now();
    _cStop = std::chrono::high_resolution_clock::now();
    cudaEventCreate(&_gStart);
    cudaEventCreate(&_gStop);
}

Timer::~Timer(){
    cudaFree(_gStart);
    cudaFree(_gStop);
    cudaEventDestroy(_gStart);
    cudaEventDestroy(_gStop);
}

void Timer::start_gpu() {
    cudaEventRecord(_gStart, 0);
}

void Timer::stop_gpu() {
    cudaEventRecord(_gStop, 0);
}

void Timer::stop_gpu(std::string msg){
    cudaEventRecord(_gStop, 0);
    char buff[100];

    CUDA_CHECK(cudaEventSynchronize(_gStart));
    CUDA_CHECK(cudaEventSynchronize(_gStop));
    cudaEventElapsedTime(&_timeElasped, _gStart, _gStop);

    sprintf(buff, "\t%-60s uses %.6lf ms", msg.c_str(), _timeElasped);
    this->_timeMsgs.emplace_back(buff);
}

void Timer::start_cpu() {
    _cStart = std::chrono::high_resolution_clock::now();
}

void Timer::stop_cpu() {
    _cStop = std::chrono::high_resolution_clock::now();
}

void Timer::show() {
    for (int i = 0; i < _timeMsgs.size(); i ++) {
        LOGV("%s", _timeMsgs[i].c_str());
    }
}

void Timer::init() {
    _timeMsgs.clear();
}

} //namespace model
