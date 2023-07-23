import torch
import torchvision
import onnxsim
import onnx
import argparse

def get_model(type, dir):
    if type == "resnet":
        model = torchvision.models.resnet50()
        file  = dir + "resnet50.onnx"
    elif type == "vgg":
        model = torchvision.models.vgg11()
        file  = dir + "vgg11.onnx"
    elif type == "mobilenet":
        model = torchvision.models.mobilenet_v3_small()
        file  = dir + "mobilenetV3.onnx"
    elif type == "efficientnet":
        model = torchvision.models.efficientnet_b0()
        file  = dir + "efficientnetb0.onnx"
    elif type == "efficientnetv2":
        model = torchvision.models.efficientnet_v2_s()
        file  = dir + "efficientnetV2.onnx"
    elif type == "regnet":
        model = torchvision.models.regnet_x_1_6gf()
        file  = dir + "regnet1.6gf.onnx"
    return model, file

def export_norm_onnx(model, file, input):
    model.cuda()
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 15)
    print("Finished normal onnx export")

    model_onnx = onnx.load(file)

    # 检查导入的onnx model
    onnx.checker.check_model(model_onnx)

    # 使用onnx-simplifier来进行onnx的简化。
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)


def main(args):
    type        = args.type
    dir         = args.dir
    input       = torch.rand(1, 3, 224, 224, device='cuda')
    model, file = get_model(type, dir)

    export_norm_onnx(model, file, input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, default="resnet")
    parser.add_argument("-d", "--dir", type=str, default="../models/")
    
    opt = parser.parse_args()
    main(opt)
