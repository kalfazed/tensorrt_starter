#include <chrono>
#include "timer.hpp"

#include "utils.hpp"
#include "cuda_runtime_api.h"

Timer::Timer(){
    _timeElasped = 0;
    _cStart = std::chrono::high_resolution_clock::now();
    _cStop = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventCreate(&_gStart));
    CUDA_CHECK(cudaEventCreate(&_gStop));
}

Timer::~Timer(){
    CUDA_CHECK(cudaEventDestroy(_gStart));
    CUDA_CHECK(cudaEventDestroy(_gStop));
}

void Timer::start_gpu() {
    CUDA_CHECK(cudaEventRecord(_gStart, 0));
}

void Timer::stop_gpu() {
    CUDA_CHECK(cudaEventRecord(_gStop, 0));
}

void Timer::stop_gpu(std::string msg){
    char buff[100];
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(_gStop, 0));
    CUDA_CHECK(cudaEventSynchronize(_gStart));
    CUDA_CHECK(cudaEventSynchronize(_gStop));
    CUDA_CHECK(cudaEventElapsedTime(&_timeElasped, _gStart, _gStop));

    sprintf(buff, "%s uses %.6lf ms", msg.c_str(), _timeElasped);
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
        LOG("%s", _timeMsgs[i].c_str());
    }
}

void Timer::init() {
    _timeMsgs.clear();
}
