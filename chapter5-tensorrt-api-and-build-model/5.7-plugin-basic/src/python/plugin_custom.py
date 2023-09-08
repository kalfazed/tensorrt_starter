import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
import struct
import os

from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

# @parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i","i", "i", "i", "none")
# def seluCustom_symbolic(
#         g,
#         input):
#     return g.op("custom::deform_conv2d", input)

# register_custom_op_symbolic("torchvision::deform_conv2d", seluCustom_symbolic, 12)

    
class Selu_customImp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        return g.op("custom::seluCustom2", x)
    
    @staticmethod
    def forward(ctx, x):
        return x / (1 + torch.exp(-x))

class Selu_custom(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return Selu_customImp.apply(x)

register_custom_op_symbolic("custom::seluCustom2", Selu_customImp.symbolic, 1)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv   = nn.Conv2d(1, 3, (3, 3))
        self.act    = Selu_custom()
        self.norm   = nn.BatchNorm2d(num_features=3)
        self.linear = nn.Linear(in_features=5, out_features=1, bias=False)

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
        x = self.norm(x)
        x = self.act(x)
        return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def export_norm_onnx(input, model):
    current_path = os.path.dirname(__file__)
    file = current_path + "/../../models/onnx/plugin_custom.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 12,
        custom_opsets= {"custom_domain": 12})
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
    
    # 导出onnx
    export_norm_onnx(input, model);

    # 计算
    eval(input, model)
