import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.autograd
import os

class MYSELUImpl(torch.autograd.Function):

    # reference: https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions
    @staticmethod
    def symbolic(g, x):
        return g.op("MYSELU", x)

    @staticmethod
    def forward(ctx, x):
        return x / (1 + torch.exp(-x))


class MYSELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return MYSELUImpl.apply(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 3, (3, 3), padding=1)
        self.myselu = MYSELU()
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.myselu(x)
        return x


# 这个包对应opset11的导出代码，如果想修改导出的细节，可以在这里修改代码
# import torch.onnx.symbolic_opset11
print("对应opset文件夹代码在这里：", os.path.dirname(torch.onnx.__file__))

model = Model().eval()
input = torch.tensor([[[
    [0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
    [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
    [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
    [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
    [0.3138, 0.1980, 0.4162, 0.2843, 0.3398]]]])

output = model(input)
print(f"inference output = \n{output}")

dummy = torch.zeros(1, 1, 5, 5)
current_path = os.path.dirname(__file__)
file = current_path + "/../../models/onnx/sample_mySelu.onnx"
torch.onnx.export(
    model, 

    # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
    (dummy,), 

    # 储存的文件路径
    file, 

    # 打印详细信息
    verbose=True, 

    # 为输入和输出节点指定名称，方便后面查看或者操作
    input_names=["image"], 
    output_names=["output"], 

    # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
    opset_version=12

    # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
    # 通常，我们只设置batch为动态，其他的避免动态
)

print("Done.!")
