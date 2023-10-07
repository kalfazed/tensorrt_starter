#include "custom-scalarAttr-plugin.hpp"
#include <map>
#include <cstring>

void customScalarAttrImpl(const float* inputs, float* outputs, const float scalar, const float scale, const int nElements, cudaStream_t stream);

using namespace nvinfer1;

namespace custom
{
REGISTER_TENSORRT_PLUGIN(CustomScalarAttrPluginCreator);

PluginFieldCollection   CustomScalarAttrPluginCreator::mFC {};
std::vector<PluginField> CustomScalarAttrPluginCreator::mAttrs;


CustomScalarAttrPlugin::CustomScalarAttrPlugin(const std::string &name, float scalar, float scale, std::string tag):
    mName(name)
{
    mParams.scalar = scalar;
    mParams.scale  = scale;
    mParams.tag    = tag;
}

CustomScalarAttrPlugin::CustomScalarAttrPlugin(const std::string &name, const void* buffer, size_t length):
    mName(name)
{
    memcpy(&mParams, buffer, sizeof(mParams));
}

CustomScalarAttrPlugin::~CustomScalarAttrPlugin()
{
    return;
}

const char* CustomScalarAttrPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomScalarAttrPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t CustomScalarAttrPlugin::getNbOutputs() const noexcept
{
    return 1;
}

size_t CustomScalarAttrPlugin::getSerializationSize() const noexcept
{
    return sizeof(mParams);
}

const char* CustomScalarAttrPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DataType CustomScalarAttrPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

DimsExprs CustomScalarAttrPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    return inputs[0];
}

size_t CustomScalarAttrPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t CustomScalarAttrPlugin::initialize() noexcept
{
    return 0;
}

void CustomScalarAttrPlugin::terminate() noexcept 
{
    return;
}

void CustomScalarAttrPlugin::serialize(void *buffer) const noexcept
{
    memcpy(buffer, &mParams, sizeof(mParams));
    return;

}

void CustomScalarAttrPlugin::destroy() noexcept
{
    delete this;
    return;
}

int32_t CustomScalarAttrPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
    const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    int nElements = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++){
        nElements *= inputDesc[0].dims.d[i];
    }

    customScalarAttrImpl(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]), 
            mParams.scalar,
            mParams.scale,
            nElements,
            stream);

    return 0;
}

IPluginV2DynamicExt* CustomScalarAttrPlugin::clone() const noexcept
{
    auto p = new CustomScalarAttrPlugin(mName, &mParams, sizeof(mParams));
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

bool CustomScalarAttrPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
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

void CustomScalarAttrPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    return;
}
void CustomScalarAttrPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
    return;
}
void CustomScalarAttrPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept 
{
    return;
}
void CustomScalarAttrPlugin::detachFromContext() noexcept 
{
    return;
}

CustomScalarAttrPluginCreator::CustomScalarAttrPluginCreator()
{
    mAttrs.emplace_back(PluginField("scalar", nullptr, PluginFieldType::kFLOAT32, 1));
    mAttrs.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mAttrs.emplace_back(PluginField("tag", nullptr, PluginFieldType::kCHAR, 0));
    mFC.nbFields = mAttrs.size();
    mFC.fields   = mAttrs.data();
}

CustomScalarAttrPluginCreator::~CustomScalarAttrPluginCreator()
{
}

const char* CustomScalarAttrPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomScalarAttrPluginCreator::getPluginVersion() const noexcept 
{
    return PLUGIN_VERSION;
}

const char* CustomScalarAttrPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV2* CustomScalarAttrPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    float scalar = 0;
    float scale = 0;
    std::string tag;

    for (int i = 0; i < fc->nbFields; i++) {
        if (strcmp(fc->fields[i].name, "scalar") == 0){
            scalar = *reinterpret_cast<const float*>(fc->fields[i].data);
        }
        if (strcmp(fc->fields[i].name, "scale") == 0){
            scale = *reinterpret_cast<const float*>(fc->fields[i].data);
        }
        if (strcmp(fc->fields[i].name, "tag") == 0){
            auto str = *reinterpret_cast<const char*>(fc->fields[i].data);
            tag = std::string(str, str + fc->fields[i].length);
        }

    }
    return new CustomScalarAttrPlugin(name, scalar, scale, tag);
    
}

IPluginV2* CustomScalarAttrPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new CustomScalarAttrPlugin(name, serialData, serialLength);
}

void CustomScalarAttrPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
    return;
}

const PluginFieldCollection* CustomScalarAttrPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

} // namespace custom
