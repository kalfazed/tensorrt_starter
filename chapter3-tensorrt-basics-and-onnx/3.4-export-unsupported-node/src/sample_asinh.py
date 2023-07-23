import torch
import torch.onnx

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = torch.asinh(x)
        return x

def infer():
    input = torch.rand(1, 5)
    
    model = Model()
    x = model(input)
    print("input is: ", input.data)
    print("result is: ", x.data)

def export_norm_onnx():
    input   = torch.rand(1, 5)
    model   = Model()
    model.eval()

    file    = "../models/sample-asinh.onnx"
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 9)
    print("Finished normal onnx export")

if __name__ == "__main__":
    infer()

    # 这里导出asinh会出现错误。
    # Pytorch可以支持asinh的同时,
    #   def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...

    # 从onnx支持的算子里面我们可以知道自从opset9开始asinh就已经被支持了
    #   asinh is suppored since opset9

    # 所以我们可以知道，问题是出现在PyTorch与onnx之间没有建立asinh的映射
    # 我们需要建立这个映射。这里涉及到了注册符号函数的概念，详细看PPT
    export_norm_onnx()
