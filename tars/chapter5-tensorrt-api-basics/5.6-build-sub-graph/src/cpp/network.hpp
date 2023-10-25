#ifndef __NETWORK_HPP__
#define __NETWORK_HPP__

#include <NvInfer.h>
#include <string>
#include <map>
#include <memory>
#include <model.hpp>

namespace network {


namespace parser {

nvinfer1::IShuffleLayer* addReshape(
    std::string layer_name,
    nvinfer1::ITensor& input, 
    std::vector<int> dims,
    std::vector<int> perm,
    nvinfer1::INetworkDefinition& network);

nvinfer1::IShuffleLayer* addPermute(
    std::string layer_name,
    nvinfer1::ITensor& input, 
    std::vector<int> perm,
    nvinfer1::INetworkDefinition& network);

nvinfer1::IFullyConnectedLayer* addFullyConnected(
    std::string layer_name,
    nvinfer1::ITensor& input, 
    int output_channel,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights);

nvinfer1::IScaleLayer* addBatchNorm(
    std::string layer_name,
    nvinfer1::ITensor& input,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights);

nvinfer1::IConvolutionLayer* addConv2d(
    std::string layer_name, 
    nvinfer1::ITensor& input,
    int kernel_size, int output_channel, int stride, int pad,
    nvinfer1::DataType prec,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights);

nvinfer1::IActivationLayer* addActivation(
    std::string layer_name,
    nvinfer1::ITensor& input, 
    nvinfer1::ActivationType type,
    nvinfer1::INetworkDefinition& network);

nvinfer1::IElementWiseLayer* addElementWise(
    std::string layer_name,
    nvinfer1::ITensor& input1, 
    nvinfer1::ITensor& input2, 
    nvinfer1::ElementWiseOperation type,
    nvinfer1::INetworkDefinition& network);

nvinfer1::IElementWiseLayer* addConvBNSiLU(
    std::string layer_name, 
    nvinfer1::ITensor& input, 
    int kernel_size, 
    int output_channel, 
    int stride, 
    int pad,
    nvinfer1::DataType prec,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights);

nvinfer1::ILayer* addC2F(
    std::string layer_name, 
    nvinfer1::ITensor& input, 
    int output_channel, 
    nvinfer1::DataType prec,
    nvinfer1::INetworkDefinition& network,
    std::map<std::string, nvinfer1::Weights> weights);


} // namespace parser

// void build_linear(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
// void build_conv(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
// void build_permute(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
// void build_reshape(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);
// void build_batchNorm(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> mWts);

void build_cbr(
    nvinfer1::INetworkDefinition& network, 
    nvinfer1::DataType prec,
    std::map<std::string, nvinfer1::Weights> weights) ;

void build_resBlock(
    nvinfer1::INetworkDefinition& network,
    nvinfer1::DataType prec,
    std::map<std::string, nvinfer1::Weights> weights) ;

void build_convBNSiLU(
    nvinfer1::INetworkDefinition& network,
    nvinfer1::DataType prec,
    std::map<std::string, nvinfer1::Weights> weights) ;

void build_C2F(
    nvinfer1::INetworkDefinition& network,
    nvinfer1::DataType prec,
    std::map<std::string, nvinfer1::Weights> weights);


}; // namespace network

#endif //__NETWORK_HPP__
