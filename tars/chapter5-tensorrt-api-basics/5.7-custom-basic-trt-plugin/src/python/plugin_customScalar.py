import torch
import torch.onnx
import torch.nn as nn
import onnxruntime
import onnx
import onnxsim
import os
from collections import OrderedDict

class CustomScalarImpl(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, r, s):
        return g.op("custom::customScalar", x, scalar_f=r, scale_f=s)

    @staticmethod
    def forward(ctx, x, r, s):
        return (x + r) * s

class CustomScalar(nn.Module):
    def __init__(self, r, s):
        super().__init__()
        self.scalar = r
        self.scale  = s

    def forward(self, x):
        return CustomScalarImpl.apply(x, self.scalar, self.scale)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv   = nn.Conv2d(1, 3, (3, 3), padding=1)
        self.act    = CustomScalar(1, 10)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=1.)
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.05)
                nn.init.constant_(m.bias, 0.05)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def export_norm_onnx(input, model):
    current_path = os.path.dirname(__file__)
    file = current_path + "/../../models/onnx/sample_customScalar.onnx"
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

def eval(input, model):
    output = model(input)
    print("------from infer------")
    print(input)
    print("\n")
    print(output)

if __name__ == "__main__":
    setup_seed(1)
    input = torch.tensor([[[
        [0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
        [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
        [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
        [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
        [0.3138, 0.1980, 0.4162, 0.2843, 0.3398]]]])

    model = Model()
    model.eval() 
    
    # 计算
    eval(input, model)

    # 导出onnx
    export_norm_onnx(input, model);


