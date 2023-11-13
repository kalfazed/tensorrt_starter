#include <iostream>
#include <memory>

#include "model.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    Model model("models/onnx/resnet50.onnx", Model::precision::FP32);

    if(!model.build()){
        LOGE("fail in building model");
        return 0;
    }

    if(!model.infer("data/fox.png")){
        LOGE("fail in infering model");
        return 0;
    }
    if(!model.infer("data/cat.png")){
        LOGE("fail in infering model");
        return 0;
    }

    if(!model.infer("data/eagle.png")){
        LOGE("fail in infering model");
        return 0;
    }

    if(!model.infer("data/gazelle.png")){
        LOGE("fail in infering model");
        return 0;
    }

    return 0;
}
