import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
import struct
import os

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, (3, 3))
        self.deconv = nn.ConvTranspose2d(3, 1, (3, 3))

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=1.)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 为了能够让TensorRT读取PyTorch导出的权重，我们可以把权重按照指定的格式导出:
# count
# [name][len][weights value in hex mode]
# [name][len][weights value in hex mode]
# ...

def export_weight(model):
    current_path = os.path.dirname(__file__)
    f = open(current_path + "/../../models/weights/sample_deconv.weights", 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))
    
    # 我们将权重里的float数据，按照hex16进制的形式进行保存，也就是所谓的编码
    # 可以使用python中的struct.pack
    for k,v in model.state_dict().items():
        print('exporting ... {}: {}'.format(k, v.shape))
        
        # 将权重转为一维
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

def export_norm_onnx(input, model):
    current_path = os.path.dirname(__file__)
    file = current_path + "/../../models/onnx/sample_deconv.onnx"
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

    input = torch.tensor([[[
        [0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
        [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
        [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
        [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
        [0.3138, 0.1980, 0.4162, 0.2843, 0.3398]]]])

    model = Model()
    
    # 以bytes形式导出权重
    export_weight(model);

    # 导出onnx
    export_norm_onnx(input, model);

    # 计算
    eval(input, model)
