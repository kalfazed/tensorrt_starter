#include <iostream>
#include <memory>

#include "model.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    Model model("models/onnx/sample.onnx");
    if(!model.build()){
        LOGE("ERROR: fail in building model");
        return 0;
    }
    return 0;
}
