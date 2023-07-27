#include "myselu-plugin.hpp"
#include "NvInfer.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;

void myselu_inference(const float* x, float* output, int n, cudaStream_t stream);

namespace custom{

const char* MYSELU_PLUGIN_VERSION{"1"};
const char* MYSELU_PLUGIN_NAME{"MYSELU"};

PluginFieldCollection MySELUPluginCreator::mFC{}; 
std::vector<PluginField> MySELUPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(MySELUPluginCreator);

template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

/***************************************************************/
/***********************Plugin本体部分**************************/
/***************************************************************/
MySELUPlugin::MySELUPlugin(const std::string name)
    : mLayerName(name)
{}

MySELUPlugin::MySELUPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char* d = static_cast<const char*>(data);
    const char* a = d;

    int nstr = readFromBuffer<int>(d);

    d += nstr;
    assert(d == (a + length));

}

const char* MySELUPlugin::getPluginType() const noexcept
{
    return MYSELU_PLUGIN_NAME;
}

const char* MySELUPlugin::getPluginVersion() const noexcept
{
    return MYSELU_PLUGIN_VERSION;
}

int MySELUPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs MySELUPlugin::getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return *inputs;
}

int MySELUPlugin::initialize() noexcept
{
    return 0;
}

int MySELUPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    
    void* output = outputs[0];
    size_t volume = 1;
    for (int i = 0; i < inputDesc->dims.nbDims; i++){
        volume *= inputDesc->dims.d[i];
    }
    mInputVolume = volume;

    myselu_inference(
        static_cast<const float*>(inputs[0]), 
        static_cast<float*>(output), 
        mInputVolume,
        stream
    );
    return 0;
}

size_t MySELUPlugin::getSerializationSize() const noexcept
{
    return sizeof(int);
}

void MySELUPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

}

bool MySELUPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{   
    auto type = inOut[pos].type;
    auto format = inOut[pos].format;
    if (type == DataType::kFLOAT && format == PluginFormat::kLINEAR)
        return true;
    else
        return false;
}

void MySELUPlugin::terminate() noexcept {}

void MySELUPlugin::destroy() noexcept
{
    delete this;
}

void MySELUPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
    const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept{

    auto type = in->desc.type;
    auto format = in->desc.format;
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kLINEAR);
}

IPluginV2DynamicExt* MySELUPlugin::clone() const noexcept
{
    auto plugin = new MySELUPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void MySELUPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MySELUPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



/***************************************************************/
/********************PluginCreator部分**************************/
/***************************************************************/

MySELUPluginCreator::MySELUPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MySELUPluginCreator::getPluginName() const noexcept
{
    return MYSELU_PLUGIN_NAME;
}

const char* MySELUPluginCreator::getPluginVersion() const noexcept
{
    return MYSELU_PLUGIN_VERSION;
}

const PluginFieldCollection* MySELUPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* MySELUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    return new MySELUPlugin(name);
}

IPluginV2* MySELUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new MySELUPlugin(name, serialData, serialLength);
}

void MySELUPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MySELUPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
} // namespace custom
