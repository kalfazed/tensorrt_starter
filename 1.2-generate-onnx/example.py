import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, weights, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        with torch.no_grad():
            self.linear.weight.copy_(weights)
    
    def forward(self, x):
        x = self.linear(x)
        return x

def infer():
    in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    weights = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ],dtype=torch.float32)
    
    model = Model(4, 3, weights)
    x = model(in_features)
    print("result is: ", x)

def export_onnx():
    input   = torch.zeros(1, 1, 1, 4)
    weights = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ],dtype=torch.float32)
    model   = Model(4, 3, weights)
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = "example.onnx",
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 12)
    print("Finished onnx export")


if __name__ == "__main__":
    infer()
    export_onnx()