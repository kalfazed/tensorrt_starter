import torch
import torch.nn as nn
import torchvision
import torch.onnx
import onnxruntime
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

# 注意
#   这里需要把args的各个参数的类型都指定
#   这里还没有实现底层对deform_conv2d的实现
#   具体dcn的底层实现是在c++完成的，这里会在后面的TensorRT plugin中回到这里继续讲这个案例
#   这里先知道对于不支持的算子，onnx如何导出即可
@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i","i", "i", "i", "none")
def dcn_symbolic(
        g,
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask):
    return g.op("custom::deform_conv2d", input, offset)

register_custom_op_symbolic("torchvision::deform_conv2d", dcn_symbolic, 12)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 3)
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)
    
    def forward(self, x):
        x = self.conv2(x, self.conv1(x))
        return x

def validate_onnx():
    input = torch.rand(1, 3, 5, 5)

    # PyTorch的推理
    model = Model()
    x     = model(input)
    print("result from Pytorch is :", x)

    # onnxruntime的推理
    sess  = onnxruntime.InferenceSession('../models/sample-deformable-conv.onnx')
    x     = sess.run(None, {'input0': input.numpy()})
    print("result from onnx is:    ", x)

def infer():
    input = torch.rand(1, 3, 5, 5)
    
    model = Model()
    x = model(input)
    print("input is: ", input.data)
    print("result is: ", x.data)

def export_norm_onnx():
    input   = torch.rand(1, 3, 5, 5)
    model   = Model()
    model.eval()

    file    = "../models/sample-deformable-conv.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 12)
    print("Finished normal onnx export")

if __name__ == "__main__":
    # infer()
    export_norm_onnx()
    validate_onnx()
