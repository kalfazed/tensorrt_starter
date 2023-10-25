#include <iostream>
#include <memory>

#include "utils.hpp"
#include "model.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    // Model model("models/onnx/sample_customScalar.onnx", Model::precision::FP32);
    Model model("models/onnx/sample_customLeakyReLU.onnx", Model::precision::FP16);
    if(!model.build()){
        LOGE("fail in building model");
        return 0;
    }
    if(!model.infer()){
        LOGE("fail in infering model");
        return 0;
    }
    return 0;
}
