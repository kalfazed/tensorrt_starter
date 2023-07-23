import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm2d(num_features=16)
        self.act1    = nn.ReLU()
        self.conv2   = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, padding=2)
        self.bn2     = nn.BatchNorm2d(num_features=64)
        self.act2    = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(in_features=64, out_features=10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x) 
        x = torch.flatten(x, 2, 3)  # B, C, H, W -> B, C, L (这一个过程产生了shape->slice->concat->reshape这一系列计算节点, 思考为什么)


        # b, c, w, h = x.shape
        # x = x.reshape(b, c, w * h)
        # x = x.view(b, c, -1)

        x = self.avgpool(x)         # B, C, L    -> B, C, 1
        x = torch.flatten(x, 1)     # B, C, 1    -> B, C
        x = self.head(x)            # B, L       -> B, 10
        return x

def export_norm_onnx():
    input   = torch.rand(1, 3, 64, 64)
    model   = Model()
    file    = "../models/sample-reshape.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 15)
    print("Finished normal onnx export")

    model_onnx = onnx.load(file)

    # 检查导入的onnx model
    onnx.checker.check_model(model_onnx)


    # 使用onnx-simplifier来进行onnx的简化。
    # 可以试试把这个简化给注释掉，看看flatten操作在简化前后的区别
    # onnx中其实会有一些constant value，以及不需要计算图跟踪的节点
    # 大家可以一起从netron中看看这些节点都在干什么

    # print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    # model_onnx, check = onnxsim.simplify(model_onnx)
    # assert check, "assert check failed"
    onnx.save(model_onnx, file)

if __name__ == "__main__":
    export_norm_onnx()
