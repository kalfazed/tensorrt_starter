import torch
import torch.nn as nn
import torch.onnx

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(num_features=16)
        self.act1  = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

def export_norm_onnx():
    input   = torch.rand(1, 3, 5, 5)
    model   = Model()
    model.eval()

    # 通过这个案例，我们一起学习一下onnx导出的时候，其实有一些节点已经被融合了
    # 思考一下为什么batchNorm不见了
    file    = "../models/sample-cbr.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 15)
    print("Finished normal onnx export")

if __name__ == "__main__":
    export_norm_onnx()
