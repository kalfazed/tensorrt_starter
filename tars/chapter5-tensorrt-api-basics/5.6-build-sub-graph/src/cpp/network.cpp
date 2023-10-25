#include "NvInfer.h"
#include "math.h"
#include "network.hpp"
#include <assert.h>
#include <utils.hpp>


using namespace std;

namespace network{
// 这里写的每一个小模块其实都可以单独拿出来封装成一个小网络使用
// 如果需要扩展，其实把这些放在perser.cpp里面


// 做一个conv + batchNorm + LeakyReLU的网络
//      conv
//       |
//       bn
//       |
//    LeakyReLU

void build_cbr(
    nvinfer1::INetworkDefinition& network, 
    nvinfer1::DataType prec,
    map<string, nvinfer1::Weights> weights) 
{
    auto input  = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 5, 5});

    auto conv   = parser::addConv2d("conv", *input, 3, 3, 1, 0, prec, network, weights);
    auto bn     = parser::addBatchNorm("norm", *conv->getOutput(0), network, weights);
    auto leaky  = parser::addActivation("leaky", *bn->getOutput(0), nvinfer1::ActivationType::kLEAKY_RELU, network);

    leaky->getOutput(0) ->setName("output0");
    network.markOutput(*leaky->getOutput(0));
}

// 做一个residual block: 
//       conv0
//       /   \
//      /    conv1
//     |      |
//     |      bn1
//     |      |
//     |     relu1
//     |      |
//     |      |
//     |     conv2
//     |      |
//     |      bn2
//      \    /
//       \  /
//       add2
//        |
//       relu2
//

void build_resBlock(
    nvinfer1::INetworkDefinition& network,
    nvinfer1::DataType prec,
    map<string, nvinfer1::Weights> weights) 
{
    auto data  = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 5, 5});

    auto conv0 = parser::addConv2d("conv0", *data, 3, 3, 1, 1, prec, network, weights);

    auto conv1 = parser::addConv2d("conv1", *conv0->getOutput(0), 3, 3, 1, 1, prec, network, weights);
    auto bn1   = parser::addBatchNorm("norm1", *conv1->getOutput(0), network, weights);
    auto relu1 = parser::addActivation("relu1", *bn1->getOutput(0), nvinfer1::ActivationType::kRELU, network);

    auto conv2 = parser::addConv2d("conv2", *relu1->getOutput(0), 3, 3, 1, 1, prec, network, weights);
    auto bn2   = parser::addBatchNorm("norm2", *conv2->getOutput(0), network, weights);

    auto add2  = parser::addElementWise("add2", *conv0->getOutput(0), *bn2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM, network);
    auto relu2 = parser::addActivation("relu2", *add2->getOutput(0), nvinfer1::ActivationType::kRELU, network);

    relu2->getOutput(0) ->setName("output0");
    network.markOutput(*relu2->getOutput(0));
}

// 做一个conv + bn + SiLU: (yolov8的模块测试)
//        conv
//         |
//         bn
//       /   \
//      |     |
//      |    sigmoid
//      \     /
//       \   /
//        Mul
//

void build_convBNSiLU(
    nvinfer1::INetworkDefinition& network,
    nvinfer1::DataType prec,
    map<string, nvinfer1::Weights> weights) 
{
    auto data  = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 5, 5});

    auto silu  = parser::addConvBNSiLU("", *data, 3, 3, 1, 1, prec, network, weights);


    silu->getOutput(0) ->setName("output0");
    network.markOutput(*silu->getOutput(0));
}


// 做一个C2F: (yolov8的模块测试)
//        input
//          |
//      convBNSiLU (n * ch)
//       /  |  \
//      /   |   \
//     |    |    |
//     |    | convBNSiLU ( 0.5n * ch)
//     |    |    |
//     |    | convBNSiLU ( 0.5n * ch)
//     |    |    |
//     |    \    /
//     |     \  /
//     |     add (0.5n * ch)
//      \    /
//       \  /
//      Concat (1.5n * ch)
//        |
//    convBNSiLU (n * ch)

void build_C2F(
    nvinfer1::INetworkDefinition& network,
    nvinfer1::DataType prec,
    map<string, nvinfer1::Weights> weights) 
{
    auto data  = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 5, 5});

    auto c2f  = parser::addC2F("", *data, 4, prec, network, weights);

    c2f->getOutput(0) ->setName("output0");
    network.markOutput(*c2f->getOutput(0));
}



} // namespace network
