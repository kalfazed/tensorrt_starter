#include <iostream>
#include <memory>

#include "utils.hpp"
#include "model.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    // Model model("models/onnx/sample.onnx");
    // Model model("models/onnx/resnet50.onnx");
    Model model("models/onnx/vgg16.onnx");

    if(!model.build()){
        LOGE("fail in building model");
        return 0;
    }
    // if(!model.infer()){
    //     LOGE("fail in infering model");
    //     return 0;
    // }
    return 0;
}
