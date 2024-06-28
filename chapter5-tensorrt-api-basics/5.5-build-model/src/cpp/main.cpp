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
     * sample_conv:           conv only:                       input shape: [1x1x5x5],     output shape: [1x3x3x3]
     * sample_permute:        conv + permute:                  input shape: [1x1x5x5],     output shape: [1x3x3x3]
     * sample_reshape:        conv + reshape + linear:         input shape: [1x1x5x5],     output shape: [1x9x3]
     * sample_batchNorm:      conv + batchNorm:                input shape: [1x1x5x5],     output shape: [1x3x3x3]
     * sample_cbr:            conv + BN + ReLU:                input shape: [1x1x5x5],     output shape: [1x1x3x3]
    */

    // Model model("models/weights/sample_linear.weights");
    // Model model("models/weights/sample_conv.weights");
    // Model model("models/weights/sample_permute.weights");
    // Model model("models/weights/sample_reshape.weights");
    // Model model("models/weights/sample_batchNorm.weights");
    // Model model("models/weights/sample_cbr.weights");
    // Model model("models/weights/sample_pooling.weights");
    // Model model("models/weights/sample_upsample.weights");
    // Model model("models/weights/sample_deconv.weights");
    // Model model("models/weights/sample_concat.weights");
    // Model model("models/weights/sample_elementwise.weights");
    // Model model("models/weights/sample_reduce.weights");
    Model model("models/weights/sample_slice.weights");

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
