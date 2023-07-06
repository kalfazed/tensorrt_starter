#include "model.hpp"
#include "NvInfer.h"
#include "math.h"
#include <memory>
#include "model.hpp"
#include "network.hpp"
#include <assert.h>
#include <utils.hpp>

using namespace std;

namespace network{

namespace parser{

// TODO... vargs
nvinfer1::IShuffleLayer* addReshape(
    string layer_name,
    nvinfer1::ITensor& input, 
    vector<int> dims,
    vector<int> perm,
    nvinfer1::INetworkDefinition& network)
{
    auto reshape = network.addShuffle(input);
    reshape->setReshapeDimensions(nvinfer1::Dims3{1, 3, -1});
    reshape->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});      
    reshape->setName(layer_name.c_str());

    return reshape;
}


// TODO... vargs
nvinfer1::IShuffleLayer* addPermute(
    string layer_name,
    nvinfer1::ITensor& input, 
    vector<int> perm,
    nvinfer1::INetworkDefinition& network)
{
    auto permute = network.addShuffle(input);
    permute->setFirstTranspose(nvinfer1::Permutation{perm[0], perm[1], perm[2], perm[3]}); // B, C, H, W -> B, H, W, C
    permute->setName(layer_name.c_str());
    return permute;
}

nvinfer1::IFullyConnectedLayer* addFullyConnected(
    string layer_name,
    nvinfer1::ITensor& input, 
    int output_channel,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights)
{
    auto fc = network.addFullyConnected(input, output_channel, weights[layer_name + ".weight"], {});
    fc->setName(layer_name.c_str());
    
    return fc;
}


nvinfer1::IScaleLayer* addBatchNorm(
    string layer_name,
    nvinfer1::ITensor& input,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights)
{
    // 因为TensorRT内部没有BatchNorm的实现，但是我们只要知道BatchNorm的计算原理，就可以使用IScaleLayer来创建BN的计算
    // IScaleLayer主要是用在quantization和dequantization，作为提前了解，我们试着使用IScaleLayer来搭建于一个BN的parser
    // IScaleLayer可以实现: y = (x * scale + shift) ^ pow

    float* gamma   = (float*)weights[layer_name + ".weight"].values;
    float* beta    = (float*)weights[layer_name + ".bias"].values;
    float* mean    = (float*)weights[layer_name + ".running_mean"].values;
    float* var     = (float*)weights[layer_name + ".running_var"].values;
    float  eps     = 1e-5;
    
    int    count   = weights[layer_name + ".running_var"].count;

    float* scales  = (float*)malloc(count * sizeof(float));
    float* shifts  = (float*)malloc(count * sizeof(float));
    float* pows    = (float*)malloc(count * sizeof(float));
    
    // 这里具体参考一下batch normalization的计算公式，网上有很多
    for (int i = 0; i < count; i ++) {
        scales[i] = gamma[i] / sqrt(var[i] + eps);
        shifts[i] = beta[i] - (mean[i] * gamma[i] / sqrt(var[i] + eps));
        pows[i]   = 1.0;
    }

    // 将计算得到的这些值写入到Weight中
    auto scales_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, scales, count};
    auto shifts_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, shifts, count};
    auto pows_weights   = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, pows, count};

    // 创建IScaleLayer并将这些weights传进去，这里使用channel作为scale model
    auto bn = network.addScale(input, nvinfer1::ScaleMode::kCHANNEL, shifts_weights, scales_weights, pows_weights);
    bn->setName(layer_name.c_str());

    LOGV("%s, %s", bn->getName(), (printDims(bn->getOutput(0)->getDimensions())).c_str());

    return bn;
}

nvinfer1::IConvolutionLayer* addConv2d(
    string layer_name, 
    nvinfer1::ITensor& input, 
    int kernel_size, 
    int output_channel, 
    int stride, 
    int pad,
    nvinfer1::DataType prec,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights)
{
    auto conv = network.addConvolutionNd(
            input, output_channel, 
            nvinfer1::DimsHW{kernel_size, kernel_size}, 
            weights[layer_name + ".weight"], 
            weights[layer_name + ".bias"]);
    conv->setName(layer_name.c_str());
    conv->setStride(nvinfer1::DimsHW(stride, stride));
    conv->setPaddingNd(nvinfer1::DimsHW(pad, pad));

    // 注意，这里setPrecision需要跟config->setFlag配合使用，否则无效
    conv->setPrecision(prec);
    LOGV("%s, %s", conv->getName(), (printDims(conv->getOutput(0)->getDimensions())).c_str());

    return conv;
}

nvinfer1::IActivationLayer* addActivation(
    string layer_name,
    nvinfer1::ITensor& input, 
    nvinfer1::ActivationType type,
    nvinfer1::INetworkDefinition& network)
{
    auto act = network.addActivation(input, type);
    act->setName(layer_name.c_str());
    LOGV("%s, %s", act->getName(), (printDims(act->getOutput(0)->getDimensions())).c_str());

    return act;

}

nvinfer1::IElementWiseLayer* addElementWise(
    string layer_name,
    nvinfer1::ITensor& input1, 
    nvinfer1::ITensor& input2,
    nvinfer1::ElementWiseOperation type,
    nvinfer1::INetworkDefinition& network)
{
    auto ew = network.addElementWise(input1, input2, type);
    ew->setName(layer_name.c_str());
    LOGV("%s, %s", ew->getName(), (printDims(ew->getOutput(0)->getDimensions())).c_str());
    
    return ew;
}
nvinfer1::IConcatenationLayer* addConcat(
    string layer_name,
    nvinfer1::ITensor* input[],
    int size,
    nvinfer1::INetworkDefinition& network)
{
    auto concat =  network.addConcatenation(input, size);
    concat->setName(layer_name.c_str());
    LOGV("%s, %s", concat->getName(), (printDims(concat->getOutput(0)->getDimensions())).c_str());

    return concat;
}

nvinfer1::ISliceLayer* addSlice(
    string layer_name,
    nvinfer1::ITensor& input, 
    nvinfer1::Dims start,
    nvinfer1::Dims size,
    nvinfer1::Dims stride,
    nvinfer1::INetworkDefinition& network)
{
    auto slice = network.addSlice(input, start, size, stride);
    slice->setName(layer_name.c_str());
    LOGV("%s, %s", slice->getName(), (printDims(slice->getOutput(0)->getDimensions())).c_str());

    return slice;
}


nvinfer1::IElementWiseLayer* addConvBNSiLU(
    string layer_name, 
    nvinfer1::ITensor& input, 
    int kernel_size, 
    int output_channel, 
    int stride, 
    int pad,
    nvinfer1::DataType prec,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights)
{
    auto conv    = addConv2d(layer_name + "conv", input, kernel_size, output_channel, stride, pad, prec, network, weights);
    auto bn      = addBatchNorm(layer_name + "norm", *conv->getOutput(0), network, weights);
    auto sigmoid = addActivation(layer_name + "sigmoid", *bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID, network);
    auto mul     = addElementWise(layer_name + "mul", *bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD, network);

    return mul;
}

nvinfer1::ILayer* addBottleNeck(
    string layer_name, 
    nvinfer1::ITensor& input, 
    int ch1, int ch2,
    bool shortcut,
    nvinfer1::DataType prec,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights)
{
    auto silu1 = addConvBNSiLU(layer_name + "cv1.", input,                3, ch1, 1, 1, prec, network, weights);
    auto silu2 = addConvBNSiLU(layer_name + "cv2.", *silu1->getOutput(0), 3, ch2, 1, 1, prec, network, weights);

    if (shortcut)  {
        auto add  =  addElementWise(layer_name + "cv1.add", 
                                    input, *silu2->getOutput(0),  
                                    nvinfer1::ElementWiseOperation::kSUM, network);
        return add;
    }
    
    return silu1;
}

// 做一个C2F: (yolov8的模块测试)
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

nvinfer1::ILayer* addC2F(
    string layer_name, 
    nvinfer1::ITensor& input, 
    int output_channel, 
    nvinfer1::DataType prec,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights)
{
    auto cv1     = addConvBNSiLU(layer_name + "cv1.", input, 1, output_channel, 1, 0, prec, network, weights);
    auto dim     = cv1->getOutput(0)->getDimensions();
    auto slice1  = addSlice(layer_name + "slice1", 
                            *cv1->getOutput(0), 
                            nvinfer1::Dims4{0,        0,          0,        0},
                            nvinfer1::Dims4{dim.d[0], dim.d[1]/2, dim.d[2], dim.d[3]},
                            nvinfer1::Dims4{1,        1,          1,        1}, 
                            network);
    auto slice2  = addSlice(layer_name + "slice2", 
                            *cv1->getOutput(0), 
                            nvinfer1::Dims4{0,        dim.d[1]/2, 0,        0}, 
                            nvinfer1::Dims4{dim.d[0], dim.d[1]/2, dim.d[2], dim.d[3]},
                            nvinfer1::Dims4{1,        1,          1,        1}, 
                            network);

    nvinfer1::ITensor* concat1Input[] = {slice1->getOutput(0), slice2->getOutput(0)};
    auto concat1 = addConcat(layer_name + "concat1", concat1Input, 2, network);

    auto add     = addBottleNeck(layer_name + "m.0.", *slice2->getOutput(0), 2, 2, true, prec, network, weights);

    nvinfer1::ITensor* concat2Input[] = {concat1->getOutput(0), add->getOutput(0)};
    auto concat2 = addConcat(layer_name + "concat2", concat2Input, 2, network);

    auto cv2     = addConvBNSiLU(layer_name + "cv2.", *concat2->getOutput(0), 1, output_channel, 1, 0, prec, network, weights);

    return cv2;
} 

}// namespace parser

} // namespace network


