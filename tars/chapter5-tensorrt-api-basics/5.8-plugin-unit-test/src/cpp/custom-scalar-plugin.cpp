#include "custom-scalar-plugin.hpp"
#include <map>
#include <cstring>

/* customScalar的核函数接口部分 */
void customScalarImpl(const float* inputs, float* outputs, const float scalar, const float scale, const int nElements, cudaStream_t stream);

using namespace nvinfer1;

namespace custom
{
/******************************************************************/
/********************注册PluginCreator*****************************/
/******************************************************************/
REGISTER_TENSORRT_PLUGIN(CustomScalarPluginCreator);

/******************************************************************/
/*********************静态变量的申明*******************************/
/******************************************************************/
PluginFieldCollection   CustomScalarPluginCreator::mFC {};
std::vector<PluginField> CustomScalarPluginCreator::mAttrs;

/******************************************************************/
/*********************CustomScalarPlugin实现部分***********************/
/******************************************************************/

CustomScalarPlugin::CustomScalarPlugin(const std::string &name, float scalar, float scale):
    mName(name)
{
    mParams.scalar = scalar;
    mParams.scale = scale;
}

CustomScalarPlugin::CustomScalarPlugin(const std::string &name, const void* buffer, size_t length):
    mName(name)
{
    memcpy(&mParams, buffer, sizeof(mParams));
}

CustomScalarPlugin::~CustomScalarPlugin()
{
    /* 这里的析构函数不需要做任何事情，生命周期结束的时候会自动调用terminate和destroy */
    return;
}

const char* CustomScalarPlugin::getPluginType() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return PLUGIN_NAME;
}

const char* CustomScalarPlugin::getPluginVersion() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return PLUGIN_VERSION;
}

int32_t CustomScalarPlugin::getNbOutputs() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return 1;
}

size_t CustomScalarPlugin::getSerializationSize() const noexcept
{
    /* 如果把所有的参数给放在mParams中的话, 一般来说所有插件的实现差不多一致 */
    return sizeof(mParams);
}

const char* CustomScalarPlugin::getPluginNamespace() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return mNamespace.c_str();
}

DataType CustomScalarPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return inputTypes[0];
}

DimsExprs CustomScalarPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    return inputs[0];
}

size_t CustomScalarPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    /* 一般来说会使用builder创建时用的workspaceSize所以这里一般什么都不做 */
    return 0;
}

int32_t CustomScalarPlugin::initialize() noexcept
{
    /* 这个一般会根据情况而定，建议每个插件都有一个自己的实现 */
    return 0;
}

void CustomScalarPlugin::terminate() noexcept 
{
    /* 
     * 这个是析构函数调用的函数。一般和initialize配对的使用
     * initialize分配多少内存，这里就释放多少内存
    */
    return;
}

void CustomScalarPlugin::serialize(void *buffer) const noexcept
{
    /* 序列化也根据情况而定，每个插件自己定制 */
    memcpy(buffer, &mParams, sizeof(mParams));
    return;

}

void CustomScalarPlugin::destroy() noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    delete this;
    return;
}

int32_t CustomScalarPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
    const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    /*
     * Plugin的核心的地方。每个插件都有一个自己的定制方案
     * Plugin直接调用kernel的地方
    */
    int nElements = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++){
        nElements *= inputDesc[0].dims.d[i];
    }

    customScalarImpl(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]), 
            mParams.scalar, 
            mParams.scale,
            nElements,
            stream);

    return 0;
}

IPluginV2DynamicExt* CustomScalarPlugin::clone() const noexcept
{
    /* 克隆一个Plugin对象，所有的插件的实现都差不多*/
    auto p = new CustomScalarPlugin(mName, &mParams, sizeof(mParams));
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

bool CustomScalarPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    /* 
     * 设置这个Plugin支持的Datatype以及TensorFormat, 每个插件都有自己的定制
     * 作为案例展示，这个customScalar插件只支持FP32，如果需要扩展到FP16以及INT8，需要在这里设置
    */
    
    switch (pos) {
    case 0:
        return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
    default:
        return false;
    }
    return false;
}

void CustomScalarPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    /* 一般不需要做任何使用，所有插件实现都差不多 */
    return;
}
void CustomScalarPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    /* 所有插件的实现都差不多 */
    mNamespace = pluginNamespace;
    return;
}
void CustomScalarPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept 
{
    /* 一般不需要做任何使用，所有插件实现都差不多 */
    return;
}
void CustomScalarPlugin::detachFromContext() noexcept 
{
    /* 一般不需要做任何使用，所有插件实现都差不多 */
    return;
}

/******************************************************************/
/*********************CustomScalarPluginCreator部分********************/
/******************************************************************/

CustomScalarPluginCreator::CustomScalarPluginCreator()
{
    /* 
     * 每个插件的Creator构造函数需要定制，主要就是获取参数以及传递参数
     * 初始化creator中的PluginField以及PluginFieldCollection
     * - PluginField::            负责获取onnx中的参数
     * - PluginFieldCollection：  负责将onnx中的参数传递给Plugin
    */

    mAttrs.emplace_back(PluginField("scalar", nullptr, PluginFieldType::kFLOAT32, 1));
    mAttrs.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mAttrs.size();
    mFC.fields   = mAttrs.data();
}

CustomScalarPluginCreator::~CustomScalarPluginCreator()
{
    /* 一般不需要做任何使用，所有插件实现都差不多 */
}

const char* CustomScalarPluginCreator::getPluginName() const noexcept
{
    /* 所有插件实现都差不多 */
    return PLUGIN_NAME;
}

const char* CustomScalarPluginCreator::getPluginVersion() const noexcept 
{
    /* 所有插件实现都差不多 */
    return PLUGIN_VERSION;
}

const char* CustomScalarPluginCreator::getPluginNamespace() const noexcept
{
    /* 所有插件实现都差不多 */
    return mNamespace.c_str();
}

IPluginV2* CustomScalarPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    /*
     * 通过Creator创建一个Plugin的实现，这个时候会通过mFC中取出需要的参数, 并实例化一个Plugin
     * 这个案例中，参数有scalar和scale两个参数。从fc中取出来对应的数据来初始化这个plugin
    */
    float scalar = 0;
    float scale  = 0;
    std::map<std::string, float*> paramMap = {{"scalar", &scalar}, {"scale", &scale}};

    for (int i = 0; i < fc->nbFields; i++) {
        if (paramMap.find(fc->fields[i].name) != paramMap.end()){
            *paramMap[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
        }
    }
    return new CustomScalarPlugin(name, scalar, scale);
}

IPluginV2* CustomScalarPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    /* 反序列化插件其实就是实例化一个插件，所有插件实现都差不多 */
    return new CustomScalarPlugin(name, serialData, serialLength);
}

void CustomScalarPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    /* 所有插件实现都差不多 */
    mNamespace = pluginNamespace;
    return;
}

const PluginFieldCollection* CustomScalarPluginCreator::getFieldNames() noexcept
{
    /* 所有插件实现都差不多 */
    return &mFC;
}

} // namespace custom
