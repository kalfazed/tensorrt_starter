#include <iostream>
#include <memory>

#include "model.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    Model model("models/resnet18.onnx");
    string imagePath = "data/tiny-cat.png";

    if(!model.build()){
        LOG("fail in building model");
        return 0;
    }
    if(!model.infer(imagePath)){
        LOG("fail in infering model");
        return 0;
    }
    return 0;
}
