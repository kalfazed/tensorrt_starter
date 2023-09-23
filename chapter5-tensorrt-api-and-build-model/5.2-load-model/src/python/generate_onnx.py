import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
import os

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=10, out_features=5, bias=False)
    
    def forward(self, x):
        x = self.linear(x)
        return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def export_norm_onnx():
    current_path = os.path.dirname(__file__)
    file = current_path + "/../../models/onnx/sample.onnx"

    input   = torch.rand(1, 10)
    model   = Model()
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 15)
    print("Finished normal onnx export")

    # check the exported onnx model
    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    # use onnx-simplifier to simplify the onnx
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)

def infer():
    setup_seed(1)
    model  = Model()
    input  = torch.tensor([[0.0193, 0.2616, 0.7713, 0.3785, 0.9980, 0.9008, 0.4766, 0.1663, 0.8045, 0.6552]])
    output = model(input)
    print(input)
    print(output)

if __name__ == "__main__":
    export_norm_onnx()
    infer()
