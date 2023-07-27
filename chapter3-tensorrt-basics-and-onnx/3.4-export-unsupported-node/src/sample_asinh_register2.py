import torch
import torch.onnx
import onnxruntime
import functools
from torch.onnx import register_custom_op_symbolic
from torch.onnx._internal import registration

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=9)

# 另外一个写法
#    这个是类似于torch/onnx/symbolic_opset*.py中的写法
#    通过torch._internal中的registration来注册这个算子，让这个算子可以与底层C++实现的aten::asinh绑定
#    一般如果这么写的话，其实可以把这个算子直接加入到torch/onnx/symbolic_opset*.py中
@_onnx_symbolic('aten::asinh')
def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = torch.asinh(x)
        return x


def validate_onnx():
    input = torch.rand(1, 5)

    # PyTorch的推理
    model = Model()
    x     = model(input)
    print("result from Pytorch is :", x)

    # onnxruntime的推理
    sess  = onnxruntime.InferenceSession('../models/sample-asinh2.onnx')
    x     = sess.run(None, {'input0': input.numpy()})
    print("result from onnx is:    ", x)

def export_norm_onnx():
    input   = torch.rand(1, 5)
    model   = Model()
    model.eval()

    file    = "../models/sample-asinh2.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 12)
    print("Finished normal onnx export")

if __name__ == "__main__":
    export_norm_onnx()

    # 自定义完onnx以后必须要进行一下验证
    validate_onnx()
