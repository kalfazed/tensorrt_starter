import torch
import torch.onnx
import onnxruntime
from torch.onnx import register_custom_op_symbolic

OperatorExportTypes = torch._C._onnx.OperatorExportTypes

class CustomOp(torch.autograd.Function):
    @staticmethod 
    def symbolic(g: torch.Graph, x: torch.Value) -> torch.Value:
        return g.op("custom_domain::customOp2", x)

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        x = x.clamp(min=0)
        return x / (1 + torch.exp(-x))

customOp = CustomOp.apply

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = customOp(x)
        return x


def validate_onnx():
    input   = torch.rand(1, 50).uniform_(-1, 1).reshape(1, 2, 5, 5)

    # PyTorch的推理
    model = Model()
    x     = model(input)
    print("result from Pytorch is :\n", x)

    # onnxruntime的推理
    sess  = onnxruntime.InferenceSession('../models/sample-customOp2.onnx')
    x     = sess.run(None, {'input0': input.numpy()})
    print("result from onnx is:    \n", x)

def export_norm_onnx():
    input   = torch.rand(1, 50).uniform_(-1, 1).reshape(1, 2, 5, 5)
    model   = Model()
    model.eval()

    file    = "../models/sample-customOp2.onnx"
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
