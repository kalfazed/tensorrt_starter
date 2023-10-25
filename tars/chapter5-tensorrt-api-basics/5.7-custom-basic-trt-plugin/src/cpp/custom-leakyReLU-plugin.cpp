#include "custom-leakyReLU-plugin.hpp"
#include "utils.hpp"
#include <map>
#include <cstring>

/******************************************************************/
/******************** CustomLeakyReLU的核函数接口部分 ****************/
/******************************************************************/
void customLeakyReLUImpl(const float* inputs, float* outputs, const float alpha, const int nElements, cudaStream_t stream);

using namespace nvinfer1;

namespace custom
{
REGISTER_TENSORRT_PLUGIN(CustomLeakyReLUPluginCreator);

PluginFieldCollection   CustomLeakyReLUPluginCreator::mFC {};
std::vector<PluginField> CustomLeakyReLUPluginCreator::mAttrs;

/******************************************************************/
/*********************CustomLeakyReLUPlugin部分*********************/
/******************************************************************/


CustomLeakyReLUPlugin::CustomLeakyReLUPlugin(const std::string &name, float alpha):
    mName(name)
{
    mParams.alpha = alpha;
    if (alpha < 0.0F) LOGE("ERROR detected when initialize plugin");
}

CustomLeakyReLUPlugin::CustomLeakyReLUPlugin(const std::string &name, const void* buffer, size_t length):
    mName(name)
{
    memcpy(&mParams, buffer, sizeof(mParams));
}

CustomLeakyReLUPlugin::~CustomLeakyReLUPlugin()
{
    return;
}

const char* CustomLeakyReLUPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomLeakyReLUPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t CustomLeakyReLUPlugin::getNbOutputs() const noexcept
{
    return 1;
}

size_t CustomLeakyReLUPlugin::getSerializationSize() const noexcept
{
    return sizeof(mParams);
}

const char* CustomLeakyReLUPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DataType CustomLeakyReLUPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

DimsExprs CustomLeakyReLUPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    return inputs[0];
}

size_t CustomLeakyReLUPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t CustomLeakyReLUPlugin::initialize() noexcept
{
    return 0;
}

void CustomLeakyReLUPlugin::terminate() noexcept 
{
    return;
}

void CustomLeakyReLUPlugin::serialize(void *buffer) const noexcept
{
    memcpy(buffer, &mParams, sizeof(mParams));
    return;

}

void CustomLeakyReLUPlugin::destroy() noexcept
{
    delete this;
    return;
}

int32_t CustomLeakyReLUPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
    const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    int nElements = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++){
        nElements *= inputDesc[0].dims.d[i];
    }

    customLeakyReLUImpl(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]), 
            mParams.alpha, 
            nElements,
            stream);

    return 0;
}

IPluginV2DynamicExt* CustomLeakyReLUPlugin::clone() const noexcept
{
    try{
        auto p = new CustomLeakyReLUPlugin(mName, &mParams, sizeof(mParams));
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const &e){
        LOGE("ERROR detected when clone plugin: %s", e.what());
    }
    return nullptr;
}

bool CustomLeakyReLUPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
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

void CustomLeakyReLUPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    return;
}
void CustomLeakyReLUPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
    return;
}
void CustomLeakyReLUPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept 
{
    return;
}
void CustomLeakyReLUPlugin::detachFromContext() noexcept 
{
    return;
}

/******************************************************************/
/*********************CustomLeakyReLUPluginCreator部分********************/
/******************************************************************/

CustomLeakyReLUPluginCreator::CustomLeakyReLUPluginCreator()
{
    mAttrs.emplace_back(PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mAttrs.size();
    mFC.fields   = mAttrs.data();
}

CustomLeakyReLUPluginCreator::~CustomLeakyReLUPluginCreator()
{
}

const char* CustomLeakyReLUPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomLeakyReLUPluginCreator::getPluginVersion() const noexcept 
{
    return PLUGIN_VERSION;
}

const char* CustomLeakyReLUPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV2* CustomLeakyReLUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    try{
        float alpha = 0;
        std::map<std::string, float*> paramMap = {{"alpha", &alpha}};

        for (int i = 0; i < fc->nbFields; i++) {
            if (paramMap.find(fc->fields[i].name) != paramMap.end()){
                *paramMap[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
            }
        }
        return new CustomLeakyReLUPlugin(name, alpha);
    }
    catch (std::exception const &e){
        LOGE("ERROR detected when create plugin: %s", e.what());
    }
    return nullptr;
}

IPluginV2* CustomLeakyReLUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    try{
        return new CustomLeakyReLUPlugin(name, serialData, serialLength);
    }
    catch (std::exception const &e){
        LOGE("ERROR detected when deserialize plugin: %s", e.what());
    }
    return nullptr;
}

void CustomLeakyReLUPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
    return;
}

const PluginFieldCollection* CustomLeakyReLUPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

} // namespace custom
