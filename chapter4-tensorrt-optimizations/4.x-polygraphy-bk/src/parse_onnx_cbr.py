import torch
import torch.nn as nn
import torch.onnx
import onnx
from parser import parse_onnx
from parser import read_weight

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(num_features=16)
        self.act1  = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


def export_norm_onnx():
    input   = torch.rand(1, 3, 5, 5)
    model   = Model()
    model.eval()

    file    = "../models/sample-cbr.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 15)
    print("Finished normal onnx export")

def main():
    export_norm_onnx()
    model = onnx.load_model("../models/sample-cbr.onnx")
    parse_onnx(model)

    initializers = model.graph.initializer
    for item in initializers:
        read_weight(item)


if __name__ == "__main__":
    main()
