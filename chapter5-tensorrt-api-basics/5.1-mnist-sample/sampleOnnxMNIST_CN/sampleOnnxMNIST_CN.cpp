#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_onnx_mnist_cn";

/* 
 * 整个案例被封装到一个类里面了, 在类里面调用创建引擎和推理的实现
 * 这个类实现了实现的隐蔽，用户通过这个类只能调用跟推理相关的函数build, infer
 */
class SampleOnnxMNIST
{
public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }
    bool build();
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; 
    nvinfer1::Dims                  mInputDims;  
    nvinfer1::Dims                  mOutputDims;
    int mNumber{0};         

    /* 使用智能指针来指向引擎，方便生命周期管理 */
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

    /* 创建网络 */
    bool constructNetwork(
        SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
        SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    bool processInput(const samplesCommon::BufferManager& buffers);
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

/*
 * 创建网络的流程基本上是这样：
 * 1. 创建一个builder
 * 2. 通过builder创建一个network
 * 3. 通过builder创建一个config
 * 4. 通过config创建一个opt(这个案例中没有)
 * 5. 对network进行创建
 *      - 可以使用parser直接将onnx中各个layer转换为trt能够识别的layer (这个案例中使用的是这个)
 *      - 也可以通过trt提供的ILayer相关的API自己从零搭建network (后面会讲)
 * 6. 序列化引擎(这个案例中没有)
 * 7. Free(如果使用的是智能指针的话，可以省去这一步)
*/
bool SampleOnnxMNIST::build()
{
    // 创建builder的时候需要传入一个logger来记录日志
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    // 在创建network的时候需要指定是implicit batch还是explicit batch
    // - implicit batch: network不明确的指定batch维度的大小, 值为0
    // - explicit batch: network明确指定batch维度的大小, 值为1
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    // IBuilderConfig是推理引擎相关的设置，比如fp16, int8, workspace size，dla这些都是在config里设置
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    // network的创建可以通过parser从onnx导出为network。注意不同layer在不同平台所对应的是不同的
    // 建议这里大家熟悉一下trt中的ILayer都有哪些。后面会用到
    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    // 为网络设置config, 以及parse
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    // 指定profile的cuda stream(平时用的不多)
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    // 通过builder来创建engine的过程，并将创建好的引擎序列化
    // 平时大家写的时候这里序列化一个引擎后会一般保存到文件里面，这个案例没有写出直接给放到一片memory中后面使用
    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    // 其实从这里以后，一般都是infer的部分。大家在创建推理引擎的时候其实写到序列化后保存文件就好了
    // 创建一个runtime来负责推理
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    // 通过runtime来把序列化后的引擎给反序列化, 当作engine来使用
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

/*
 * 创建network的过程，如果不是使用parser的话，需要自己一层一层的搭建。后面会讲
*/
bool SampleOnnxMNIST::constructNetwork(
    SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
    SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

/*
 * 推理的实现部分。注意这里面把反序列化的部分给省去了。直接从创建context开始
 * 这里面稍微说明一下这里的context。context就是上下文，用来创建一些空间来存储一些中间值。通过engine来创建
 * 一个engine可以创建多个context，用来负责多个不同的推理任务。
 * 另外context可以复用。也就是每次新的推理可以利用之前创建好的context
 *
 * 这个sample给提供的infer的实现非常simple, 主要在于BufferManager的实现
 * 这个BufferManager是基于RAII(Resource Acquisition Is Initialization)的设计思想建立的，
 * 方面我们在管理CPU和GPU上的buffer的使用。让整个代码变得很简洁和可读性高。
 * 不懂RAII的同学借这个机会学习一下这个，后面会用到
*/
bool SampleOnnxMNIST::infer()
{
    // 这个BufferManager类的对象buffers在创建的初期就已经帮我们把engine推理所需要的host/deivce memory已经分配好了
    // 否则我们需要自己计算engine的input/output的维度和大小，
    // 以及根据这些维度和大小进行malloc或者cudaMalloc这种内存分配
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // 对于MNIST数据的preprocess(预处理)部分, 这个案例是在CPU上实现的
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // 将host上预处理好的数据copy到device上
    buffers.copyInputToDevice();

    // 进行TensorRT的forward推理实现
    // 创建好了context之后，推理只需要使用executeV2或者enqueueV2就可以了
    // 之后trt会自动根据创建好的engine来逐层进行forward
    //  - enqueue, enqueueV2是异步推理。V2代表explicit batch
    //  - execute, executeV2是同步推理。V2代表explicit batch
    //  现在一般用的都是enqueueV2
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // 将device上forward好的数据copy到host上
    buffers.copyOutputToHost();

    // postprocess(后处理的实现)
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

/*
 * MNIST前处理的实现(这里不详细讲了，主要看一下流程)
 *   - 读取随机digit数据
 *   - 分配buffers中的host上的空间
 *   - 将数据转为浮点数
*/
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    mNumber = rand() % 10;
    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print an ascii representation
    sample::gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}

/*
 * MNIST后处理的实现(同样，这里不详细讲了，主要看一下流程)
 *   - 分配输出所需要的空间
 *   - 手动实现一个cpu版本的softmax
 *   - 输出最大值以及所对应的digit class
*/
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
                         << std::endl;
    }
    sample::gLogInfo << std::endl;

    return idx == mNumber && val > 0.9f;
}

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

/*
 * 整个main写的比较精简。整体上通过SampleOnnxMNIST这个类把很多底层的实现部分给隐藏了
 * 我们在main中所关注的只是
 *   - "拿到一个onnx"
 *   - "parse这个onnx来生成trt推理引擎",
 *   - "推理"
 *   - "打印输出"
 * 所以程序的设计也需要把与这些不相关的不要暴露在外面。提高代码的可读性
 * 这个课程后面的代码也基本按照这个思路设计
*/
int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    // 创建一个logger用来保存日志。
    // 这里需要注意一点，日志一般都是继承nvinfer1::ILogger来实现一个自定义的。
    // 由于ILogger有一些函数都是虚函数，所以我们需要自己设计
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(sampleTest);

    // 创建sample对象，只暴露build和infer接口
    SampleOnnxMNIST sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    // 创建推理引擎
    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    // 推理
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
