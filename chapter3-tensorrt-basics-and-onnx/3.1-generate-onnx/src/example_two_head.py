import torch
import torch.nn as nn
import torch.onnx

class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, weights1, weights2, bias=False):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features, bias)
        self.linear2 = nn.Linear(in_features, out_features, bias)
        with torch.no_grad():
            self.linear1.weight.copy_(weights1)
            self.linear2.weight.copy_(weights2)

    
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        return x1, x2

def infer():
    in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    weights1 = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ],dtype=torch.float32)
    weights2 = torch.tensor([
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ],dtype=torch.float32)
    
    model = Model(4, 3, weights1, weights2)
    x1, x2 = model(in_features)
    print("result is: \n")
    print(x1)
    print(x2)

def export_onnx():
    input    = torch.zeros(1, 1, 1, 4)
    weights1 = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ],dtype=torch.float32)
    weights2 = torch.tensor([
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ],dtype=torch.float32)
    model   = Model(4, 3, weights1, weights2)
    model.eval() #添加eval防止权重继续更新

    # pytorch导出onnx的方式，参数有很多，也可以支持动态size
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = "../models/example_two_head.onnx",
        input_names   = ["input0"],
        output_names  = ["output0", "output1"],
        opset_version = 12)
    print("Finished onnx export")


if __name__ == "__main__":
    infer()
    export_onnx()
