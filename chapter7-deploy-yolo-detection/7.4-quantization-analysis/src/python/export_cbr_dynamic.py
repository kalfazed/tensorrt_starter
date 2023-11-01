import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
import struct

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv   = nn.Conv2d(1, 3, (3, 3))
        self.act    = nn.LeakyReLU()
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
        # 这里由于weight的初始化是kaiming_normal_，已经达到了标准化了
        # 为了体现BN能够发生改变，将BN的weight和bias都做加1处理
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
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
    f = open("../models/sample_cbr.dynamic", 'w')
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
    file = "../models/sample_cbr_dynamic.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 15,
        dynamic_axes  = {'input0':  {0: "batch"},
                         'output0': {0: "batch"}
                         })
    print("Finished normal onnx export")

    # check the exported onnx model
    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    # use onnx-simplifier to simplify the onnx
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model = model_onnx)

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
    # 注意，这里有个坑，建议把eval注释掉看看不同
    # 这里需要在export之前进行eval，防止BN层不更新。否则BN层的权重会更新，如果模型中有Dropout也会如此
    # 推荐在以后的导出，以及推理以前都进行eval来固定权重
    
    # 以bytes形式导出权重
    export_weight(model);


    # 导出onnx
    export_norm_onnx(input, model);

    # 计算
    eval(input, model)
