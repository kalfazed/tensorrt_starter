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
    
def hook_forward(fn):
    fnnames   = fn.split(".")
    fn_module = eval(".".join(fnnames[:-1]))
    fn_name   = fnnames[-1]
    oldfn     = getattr(fn_module, fn_name)

    def make_hook(bind_fn):
        def myforward(self, x):
            y = oldfn(self, x).clone()
            bind_fn(self, x, y)
            return y

        setattr(fn_module, fn_name, myforward)
    return make_hook

@hook_forward("torch.nn.Conv2d.forward")
def symbolic_conv2d(self, x, y):
    print(f"{type(self)}: {x.shape} -> {y.shape}")
    print(f"input : {x}")
    print(f"output : {y}")

@hook_forward("torch.nn.SiLU.forward")
def symbolic_silu(self, x, y):
    print(f"{type(self)}: {x.shape} -> {y.shape}")
    print(f"input : {x}")
    print(f"output : {y}")
    
if __name__ == "__main__":
    model = Model()
    x = torch.zeros(1, 3, 3, 3)
    y = model(x)
    print(y)

