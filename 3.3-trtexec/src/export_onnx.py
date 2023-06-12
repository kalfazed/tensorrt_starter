import torch
import torchvision
import onnxsim
import onnx

def export_norm_onnx():
    model   = torchvision.models.vgg16()
    input   = torch.rand(1, 3, 224, 224, device='cuda')
    file    = "../models/vgg16.onnx"
    model.cuda()
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


if __name__ == "__main__":
    export_norm_onnx()
