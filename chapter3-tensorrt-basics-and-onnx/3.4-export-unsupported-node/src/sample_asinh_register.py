import torch
import torch.onnx
import onnxruntime
from torch.onnx import register_custom_op_symbolic

# 创建一个asinh算子的symblic，符号函数，用来登记
# 符号函数内部调用g.op, 为onnx计算图添加Asinh算子
#   g: 就是graph，计算图
#   也就是说，在计算图中添加onnx算子
#   由于我们已经知道Asinh在onnx是有实现的，所以我们只要在g.op调用这个op的名字就好了
#   symblic的参数需要与Pytorch的asinh接口函数的参数对齐
#       def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...
def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

# 在这里，将asinh_symbolic这个符号函数，与PyTorch的asinh算子绑定。也就是所谓的“注册算子”
# asinh是在名为aten的一个c++命名空间下进行实现的

# 那么aten是什么呢？
# aten是"a Tensor Library"的缩写，是一个实现张量运算的C++库
register_custom_op_symbolic('aten::asinh', asinh_symbolic, 12)


# 这里容易混淆的地方：
# 1. register_op中的第一个参数是PyTorch中的算子名字: aten::asinh
# 2. g.op中的第一个参数是onnx中的算子名字: Asinh

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
    sess  = onnxruntime.InferenceSession('../models/sample-asinh.onnx')
    x     = sess.run(None, {'input0': input.numpy()})
    print("result from onnx is:    ", x)

def export_norm_onnx():
    input   = torch.rand(1, 5)
    model   = Model()
    model.eval()

    file    = "../models/sample-asinh.onnx"
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
