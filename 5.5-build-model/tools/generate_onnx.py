import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv   = nn.Conv2d(1, 1, (3, 3))
        # self.linear = nn.Linear(in_features=5, out_features=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=1.)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.wdight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv(x)
        # x = self.linear(x)
        return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def export_norm_onnx(input, model):
    file = "../models/sample_linear.onnx"
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
    print(output)


if __name__ == "__main__":
    setup_seed(1)

    input = torch.tensor([[[0.0193, 0.2616, 0.7713, 0.3785, 0.9980],
                           [0.0193, 0.2616, 0.7713, 0.3785, 0.9980],
                           [0.0193, 0.2616, 0.7713, 0.3785, 0.9980],
                           [0.0193, 0.2616, 0.7713, 0.3785, 0.9980],
                           [0.0193, 0.2616, 0.7713, 0.3785, 0.9980]]])
    model = Model()

    export_norm_onnx(input, model)
    eval(input, model)
