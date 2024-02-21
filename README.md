CUDA and TensorRT Starter Workspace
===

This repository guides freshmen who does not have background of parallel programming in C++ to learn CUDA and TensorRT from the beginning.

- [CUDA and TensorRT Starter Workspace](#cuda-and-tensorrt-starter-workspace)
  - [How to install](#how-to-install)
  - [How to run](#how-to-run)
  - [Chapter description](#chapter-description)
    - [chapter1-build-environment](#chapter1-build-environment)
    - [chapter2-cuda-programming](#chapter2-cuda-programming)
    - [chapter3-tensorrt-basics-and-onnx](#chapter3-tensorrt-basics-and-onnx)
    - [chapter4-tensorrt-optimiztion](#chapter4-tensorrt-optimiztion)
    - [chapter5-tensorrt-api-basics](#chapter5-tensorrt-api-basics)
    - [chapter6-deploy-classification-and-inference-design](#chapter6-deploy-classification-and-inference-design)
    - [chapter7-deploy-yolo-detection](#chapter7-deploy-yolo-detection)

This repository is still working in progress(~24/02/21). I will add some more samples and more detailed description in the future. Please feel free to contribute to this repository

## How to install
Please pull the repository firstly
```shell
git clone git@github.com:kalfazed/tensorrt_starter.git
```
After clone the repository, please modify the opencv, cuda, cudnn, and TensorRT version and install directory in `config/Makefile.config` located in the root direcoty of the repository. The recommaned version in this repository is `opencv==4.x, cuda==11.6, cudnn==8.9, TensorRT==8.6.1.6`

```Makefile
# Please change the cuda version if needed
# In default, cuDNN library is located in /usr/local/cuda/lib64
CXX                         :=  g++
CUDA_VER                    :=  11

# Please modify the opencv and tensorrt install directory
OPENCV_INSTALL_DIR          :=  /usr/local/include/opencv4
TENSORRT_INSTALL_DIR        :=  /home/kalfazed/packages/TensorRT-8.6.1.6
```

Besides, please also change the `ARCH` in `config/Makefile.config`. This parameter will be used by `nvcc`, which is a compiler for cuda program. 

## How to run
Inside each subfolder of each chapter, the basic directory structure is as follow: (For some chapters, it will be different)
```shell
|-config
    |- Makefile.config
|-src
    |- cpp
        |- xxx.c
    |- python
        |- yyy.py
|-Makefile
```
Please run `make` firstly, then it will generate a binary named `trt-cuda` or `trt-infer`, depending on different chapters. Pleae run the binary directly or run `make run` command. 


## Chapter description
### chapter1-build-environment
- [1.0-build-environment](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter1-build-environment/1.0-build-environment)
### chapter2-cuda-programming
- [2.1-dim_and_index](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.1-dim_and_index)
- [2.2-cpp_cuda_interactive](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.2-cpp_cuda_interactive)
- [2.3-matmul-basic](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.3-matmul-basic)
- [2.4-error-handler](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.4-error-handler)
- [2.5-device-info](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.5-device-info)
- [2.6-nsight-system-and-compute](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.6-nsight-system-and-compute)
- [2.7-matmul-shared-memory](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.7-matmul-shared-memory)
- [2.8-bank-conflict](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.8-bank-conflict)
- [2.9-stream-and-event](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.9-stream-and-event)
- [2.10-bilinear-interpolation](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.10-bilinear-interpolation)
- [2.11-bilinear-interpolation-template](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.11-bilinear-interpolation-template)
- [2.12-affine-transformation](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.12-affine-transformation)
- [2.13-implicit-gemm-conv](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.12-implicit-gemm-conv)
- [2.14-pcd-voxelization](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter2-cuda-programming/2.13-pcd-voxelization)
### chapter3-tensorrt-basics-and-onnx
- [3.1-generate-onnx](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter3-tensorrt-basics-and-onnx/3.1-generate-onnx)
- [3.2-export-onnx](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter3-tensorrt-basics-and-onnx/3.2-export-onnx)
- [3.3-read-and-parse-onnx](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter3-tensorrt-basics-and-onnx/3.3-read-and-parse-onnx)
- [3.4-export-unsupported-node](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter3-tensorrt-basics-and-onnx/3.4-export-unsupported-node)
- [3.5-onnxsurgeon](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter3-tensorrt-basics-and-onnx/3.5-onnxsurgeon)
- [3.6-export-onnx-from-oss](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter3-tensorrt-basics-and-onnx/3.6-export-onnx-from-oss)
- [3.7-trtexec-analysis](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter3-tensorrt-basics-and-onnx/3.7-trtexec-analysis)
### chapter4-tensorrt-optimiztion
- [4.1-polygraphy[WIP]](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter4-tensorrt-optimizations/4.x-polygraphy)
### chapter5-tensorrt-api-basics
- [5.1-mnist-sample](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter5-tensorrt-api-basics/5.1-mnist-sample)
- [5.2-load-model](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter5-tensorrt-api-basics/5.2-load-model)
- [5.3-infer-model](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter5-tensorrt-api-basics/5.3-infer-model)
- [5.4-print-structure](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter5-tensorrt-api-basics/5.4-print-structure)
- [5.5-build-model](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter5-tensorrt-api-basics/5.5-build-model)
- [5.6-build-sub-graph](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter5-tensorrt-api-basics/5.6-build-sub-graph)
- [5.7-custom-basic-trt-plugin](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter5-tensorrt-api-basics/5.7-custom-basic-trt-plugin)
- [5.8-plugin-unit-test](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter5-tensorrt-api-basics/5.8-plugin-unit-test)
### chapter6-deploy-classification-and-inference-design
- [6.0-preprocess-speed-compare](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter6-deploy-classification-and-inference-design/6.0-preprocess-speed-compare)
- [6.1-deploy-classification](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter6-deploy-classification-and-inference-design/6.1-deploy-classification)
- [6.2-deploy-classification-advanced](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter6-deploy-classification-and-inference-design/6.2-deploy-classification-advanced)
- [6.3-int8-calibration](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter6-deploy-classification-and-inference-design/6.3-in8-calibration)
- [6.4-trt-engine-inspector](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter6-deploy-classification-and-inference-design/6.4-trt-engine-inspector)
### chapter7-deploy-yolo-detection
- [7.1-load-save-tensor](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter7-deploy-yolo-detection/7.1-load-save-tensor)
- [7.2-affine-transformation](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter7-deploy-yolo-detection/7.2-affine-transformation)
- [7.3-deploy-yolo-basic](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter7-deploy-yolo-detection/7.3-deploy-yolo-basic)
- [7.4-quantization-analysis](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter7-deploy-yolo-detection/7.4-quantization-analysis)
- [7.5-deploy-yolo-multitask](https://github.com/kalfazed/tensorrt_starter/tree/main/chapter7-deploy-yolo-detection/7.5-deploy-yolo-multitask)