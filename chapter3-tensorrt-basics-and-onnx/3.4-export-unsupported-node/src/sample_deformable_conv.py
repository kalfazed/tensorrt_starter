import torch
import torch.nn as nn
import torchvision
import torch.onnx

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 3)
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)
    
    def forward(self, x):
        x = self.conv2(x, self.conv1(x))
        return x

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
    infer()

    # 这里导出deformable-conv会出现错误。
    # torchvision支持deformable_conv的
    # 但是我们在onnx中是没有找到有关deformable conv的支持
    # 所以这个时候，我们需要做两件事情
    export_norm_onnx()
