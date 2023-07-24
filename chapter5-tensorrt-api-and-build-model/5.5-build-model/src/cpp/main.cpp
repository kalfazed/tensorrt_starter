#include <iostream>
#include <memory>

#include "utils.hpp"
#include "model.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    /*
     * 这里面依次举几个例子来进行展示, 对应的输入和输出也会不一样
     * sample_linear:         linear only:                     input shape: [1x5],         output shape: [1]
     * sample_cbr:            conv + BN + ReLU:                input shape: [1x1x5x5],     output shape: [1x1x3x3]
     * sample_reshape:        cbr + reshape + linear:          input shape: [1x3x5x5],     output shape: [1]
     * sample_vgg:            vgg:                             input shape: [1x3x224x224], output shape: [1x1000]
     * sample_resnet:         resnet:                          input shape: [1x3x224x224], output shape: [1x1000]
    */
    // Model model("models/weights/sample_linear.weights");
    // Model model("models/weights/sample_conv.weights");
    // Model model("models/weights/sample_permute.weights");
    Model model("models/weights/sample_reshape.weights");         //目前观测到reshape会有精度下降
    // Model model("models/weights/sample_batchNorm.weights");
    // Model model("models/weights/sample_cbr.weights");

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
