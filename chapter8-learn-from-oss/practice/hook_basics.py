import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 3, 1, 1)
        self.silu1 = nn.SiLU()
        self.conv2 = nn.Conv2d(3, 1, 1, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.silu1(x)
        x = self.conv2(x)
        return x
    
def hook_forward(oldfn):
    def myforward(self, x):
        y = oldfn(self, x)
        print(f"{type(self)}: {x.shape} -> {y.shape}")
        print(f"input : {x}")
        print(f"output : {y}")
        print("\n")
        return y
    return myforward

# 这里实现了一个基本的hook函数，当在调用conv2d和relu的forward的时候，会跳到这里来实现
# 这是一个基本的hook功能，也就是替代函数的实现

nn.Conv2d.forward = hook_forward(nn.Conv2d.forward)
nn.SiLU.forward   = hook_forward(nn.SiLU.forward)

if __name__ == "__main__":
    model = Model()
    x = torch.zeros(1, 3, 3, 3)
    y = model(x)
    print(y)

