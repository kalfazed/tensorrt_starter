import torch
import torchvision
import onnxsim
import onnx
import argparse
import struct

def get_model(type, dir):
    if type == "resnet":
        model = torchvision.models.resnet18(pretrained=True)
        name  = "resnet18"
    elif type == "vgg":
        model = torchvision.models.vgg11(pretrained=True)
        name  =  "vgg11"

    prefix = dir + name
    onnx_path = prefix + ".onnx"
    pth_path  = prefix + ".pth"

    torch.save(model, pth_path)

    return model, onnx_path, prefix

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

def export_pth_weight(prefix):
    net = torch.load(prefix + ".pth")
    f   = open(prefix + ".weights", 'w')

    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")


def main(args):
    type        = args.type
    dir         = args.dir
    input       = torch.rand(1, 3, 224, 224, device='cuda')

    model, onnx_path, prefix = get_model(type, dir)

    # export_norm_onnx(model, file, input)
    export_pth_weight(prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, default="resnet")
    parser.add_argument("-d", "--dir", type=str, default="../models/")
    
    opt = parser.parse_args()
    main(opt)
