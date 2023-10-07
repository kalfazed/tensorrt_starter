#include "custom-cublasSgemm-plugin.hpp"
#include "utils.hpp"
#include "cublas_v2.h"
#include <map>
#include <cstring>

/******************************************************************/
/******************** CustomCublasSgemm的核函数接口部分 ****************/
/******************************************************************/
// TODO
void customCublasSgemmImpl(
    const float* inputs, float* outputs, 
    const float* weights, 
    const int m, const int k, const int n,
    const int nElements, 
    cublasHandle_t handler,
    cudaStream_t stream);

using namespace nvinfer1;

namespace custom
{
REGISTER_TENSORRT_PLUGIN(CustomCublasSgemmPluginCreator);

PluginFieldCollection   CustomCublasSgemmPluginCreator::mFC {};
std::vector<PluginField> CustomCublasSgemmPluginCreator::mAttrs;

/******************************************************************/
/*********************CustomCublasSgemmPlugin部分*********************/
/******************************************************************/


CustomCublasSgemmPlugin::CustomCublasSgemmPlugin(const std::string &name, float* weight):
    mName(name)
{
    // TODO
    // mParams.alpha = alpha;
    // if (alpha < 0.0F) LOGE("ERROR detected when initialize plugin");
}

CustomCublasSgemmPlugin::CustomCublasSgemmPlugin(const std::string &name, const void* buffer, size_t length):
    mName(name)
{
    memcpy(&mParams, buffer, sizeof(mParams));
}

CustomCublasSgemmPlugin::~CustomCublasSgemmPlugin()
{
    return;
}

const char* CustomCublasSgemmPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomCublasSgemmPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t CustomCublasSgemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

size_t CustomCublasSgemmPlugin::getSerializationSize() const noexcept
{
    // TODO
    return sizeof(mParams);
}

const char* CustomCublasSgemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DataType CustomCublasSgemmPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

DimsExprs CustomCublasSgemmPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    return inputs[0];
}

size_t CustomCublasSgemmPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t CustomCublasSgemmPlugin::initialize() noexcept
{
    return 0;
}

void CustomCublasSgemmPlugin::terminate() noexcept 
{
    return;
}

void CustomCublasSgemmPlugin::serialize(void *buffer) const noexcept
{
    // TODO
    memcpy(buffer, &mParams, sizeof(mParams));
    return;

}

void CustomCublasSgemmPlugin::destroy() noexcept
{
    delete this;
    return;
}

int32_t CustomCublasSgemmPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
    const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    // TODO
    // int nElements = 1;
    // for (int i = 0; i < inputDesc[0].dims.nbDims; i++){
    //     nElements *= inputDesc[0].dims.d[i];
    // }

    // customCublasSgemmImpl(
    //         static_cast<const float*>(inputs[0]),
    //         static_cast<float*>(outputs[0]), 
    //         mParams.alpha, 
    //         nElements,
    //         stream);

    return 0;
}

IPluginV2DynamicExt* CustomCublasSgemmPlugin::clone() const noexcept
{
    try{
        auto p = new CustomCublasSgemmPlugin(mName, &mParams, sizeof(mParams));
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const &e){
        LOGE("ERROR detected when clone plugin: %s", e.what());
    }
    return nullptr;
}

bool CustomCublasSgemmPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    
    // TODO
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

void CustomCublasSgemmPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    return;
}
void CustomCublasSgemmPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
    return;
}
void CustomCublasSgemmPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept 
{
    return;
}
void CustomCublasSgemmPlugin::detachFromContext() noexcept 
{
    return;
}

/******************************************************************/
/*********************CustomCublasSgemmPluginCreator部分********************/
/******************************************************************/

CustomCublasSgemmPluginCreator::CustomCublasSgemmPluginCreator()
{
    // TODO
    // mAttrs.emplace_back(PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 1));
    // mFC.nbFields = mAttrs.size();
    // mFC.fields   = mAttrs.data();
}

CustomCublasSgemmPluginCreator::~CustomCublasSgemmPluginCreator()
{
}

const char* CustomCublasSgemmPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* CustomCublasSgemmPluginCreator::getPluginVersion() const noexcept 
{
    return PLUGIN_VERSION;
}

const char* CustomCublasSgemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV2* CustomCublasSgemmPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    // TODO
    // try{
    //     float alpha = 0;
    //     std::map<std::string, float*> paramMap = {{"alpha", &alpha}};

    //     for (int i = 0; i < fc->nbFields; i++) {
    //         if (paramMap.find(fc->fields[i].name) != paramMap.end()){
    //             *paramMap[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
    //         }
    //     }
    //     return new CustomCublasSgemmPlugin(name, alpha);
    // }
    // catch (std::exception const &e){
    //     LOGE("ERROR detected when create plugin: %s", e.what());
    // }
    return nullptr;
}

IPluginV2* CustomCublasSgemmPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    try{
        return new CustomCublasSgemmPlugin(name, serialData, serialLength);
    }
    catch (std::exception const &e){
        LOGE("ERROR detected when deserialize plugin: %s", e.what());
    }
    return nullptr;
}

void CustomCublasSgemmPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
    return;
}

const PluginFieldCollection* CustomCublasSgemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

} // namespace custom
