import torch
import torch.nn as nn
import torch.onnx
import struct

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

# 为了能够让TensorRT读取PyTorch导出的权重，我们可以把权重按照指定的格式导出:
# count
# [name][len][weights value in hex mode]
# [name][len][weights value in hex mode]
# ...

def export_weight(model):
    f = open("../models/sample_linear.weights", 'w')
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
        print("weight is {}".format(vr))


if __name__ == "__main__":
    setup_seed(1)

    input = torch.tensor([[[0.0193, 0.2616, 0.7713, 0.3785, 0.9980],
                           [0.0193, 0.2616, 0.7713, 0.3785, 0.9980],
                           [0.0193, 0.2616, 0.7713, 0.3785, 0.9980],
                           [0.0193, 0.2616, 0.7713, 0.3785, 0.9980],
                           [0.0193, 0.2616, 0.7713, 0.3785, 0.9980]]])
    model = Model()

    export_weight(model);
