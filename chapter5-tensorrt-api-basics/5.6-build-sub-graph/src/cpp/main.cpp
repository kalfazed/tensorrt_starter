#include <iostream>
#include <memory>

#include "utils.hpp"
#include "model.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    /*
     * 这里面依次举几个例子来进行展示, 对应的输入和输出也会不一样
     * sample_cbr:            conv + BN + ReLU:                input shape: [1x1x5x5],     output shape: [1x1x3x3]
     * sample_resBlock:       ---:                             input shape: [1x1x5x5],     output shape: [1x3x5x5]
     * sample_convBNSiLU:     conv + BN + SeLU:                input shape: [1x1x5x5],     output shape: [1x3x5x5]
     * sample_c2f:            ---:                             input shape: [1x1x5x5],     output shape: [1x4x5x5]
    */
    
    // Model model("models/weights/sample_cbr.weights", Model::precision::FP32);
    // Model model("models/weights/sample_cbr.weights", Model::precision::FP16);
    // Model model("models/weights/sample_resBlock.weights", Model::precision::FP32);
    // Model model("models/weights/sample_resBlock.weights", Model::precision::FP16);
    // Model model("models/weights/sample_convBNSiLU.weights", Model::precision::FP32);
    // Model model("models/weights/sample_c2f.weights", Model::precision::FP32);
    Model model("models/weights/sample_c2f.weights", Model::precision::FP16);

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
