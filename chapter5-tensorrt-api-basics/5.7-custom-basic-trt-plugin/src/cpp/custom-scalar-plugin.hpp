#ifndef __CUSTOM_SCARLAR_PLUGIN_HPP
#define __CUSTOM_SCARLAR_PLUGIN_HPP

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include <NvInfer.h>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace custom 
{
static const char* PLUGIN_NAME {"customScalar"};
static const char* PLUGIN_VERSION {"1"};


/* 
* 在这里面需要创建两个类, 一个是普通的Plugin类, 一个是PluginCreator类
*  - Plugin类是插件类，用来写插件的具体实现
*  - PluginCreator类是插件工厂类，用来根据需求创建插件。调用插件是从这里走的
*/

class CustomScalarPlugin : public IPluginV2DynamicExt {
public:
    /*
     * 我们在编译的过程中会有大概有三次创建插件实例的过程
     * 1. parse阶段: 第一次读取onnx来parse这个插件。会读取参数信息并转换为TensorRT格式
     * 2. clone阶段: parse完了以后，TensorRT为了去优化这个插件会复制很多副本出来来进行很多优化测试。也可以在推理的时候供不同的context创建插件的时候使用
     * 3. deseriaze阶段: 将序列化好的Plugin进行反序列化的时候也需要创建插件的实例
    */
    CustomScalarPlugin() = delete; //默认构造函数，一般直接delete
    CustomScalarPlugin(const std::string &name, float scalar, float scale);  //parse, clone时候用的构造函数
    CustomScalarPlugin(const std::string &name, const void* buffer, size_t length); //反序列化的时候用的构造函数

    ~CustomScalarPlugin();

    /* 有关获取plugin信息的方法 */
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    size_t      getSerializationSize() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    DataType    getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    DimsExprs   getOutputDimensions(int32_t outputIndex, const DimsExprs* input, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    size_t      getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;

    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    int32_t     enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* ionputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override; // 实际插件op执行的地方，具体实现forward的推理的CUDA/C++实现会放在这里面
    IPluginV2DynamicExt* clone() const noexcept override;

    bool        supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOuts, int32_t nbInputs, int32_t nbOutputs) noexcept override; //查看pos位置的索引是否支持指定的DataType以及TensorFormat
    void        configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override; //配置插件，一般什么都不干
    void        setPluginNamespace(const char* pluginNamespace) noexcept override;

    void        attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void        detachFromContext() noexcept override;

private:
    const std::string mName;
    std::string       mNamespace;
    struct {
        float scalar;
        float scale;
    } mParams; // 当这个插件op需要有参数的时候，把这些参数定义为成员变量，可以单独拿出来定义，也可以像这样定义成一个结构体
};

class CustomScalarPluginCreator : public IPluginCreator {
public:
    CustomScalarPluginCreator();  //初始化mFC以及mAttrs
    ~CustomScalarPluginCreator();

    const char*                     getPluginName() const noexcept override;
    const char*                     getPluginVersion() const noexcept override;
    const PluginFieldCollection*    getFieldNames() noexcept override;
    const char*                     getPluginNamespace() const noexcept override;
    IPluginV2*                      createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;  //通过包含参数的mFC来创建Plugin。调用上面的Plugin的构造函数
    IPluginV2*                      deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void                            setPluginNamespace(const char* pluginNamespace) noexcept override;
      
private:
    static PluginFieldCollection    mFC;           //接受plugionFields传进来的权重和参数，并将信息传递给Plugin，内部通过createPlugin来创建带参数的plugin
    static std::vector<PluginField> mAttrs;        //用来保存这个插件op所需要的权重和参数, 从onnx中获取, 同样在parse的时候使用
    std::string                     mNamespace;
    
};

} // namespace custom

#endif __CUSTOM_SCARLAR_PLUGIN_HPP
