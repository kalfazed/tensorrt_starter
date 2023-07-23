import numpy as np
import onnx
from onnx import numpy_helper


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    initializer = onnx.helper.make_tensor(
        name      = name,
        data_type = data_type,
        dims      = tensor_array.shape,
        vals      = tensor_array.flatten().tolist())

    return initializer


def main():
    
    input_batch    = 1;
    input_channel  = 3;
    input_height   = 64;
    input_width    = 64;
    output_channel = 16;

    input_shape    = [input_batch, input_channel, input_height, input_width]
    output_shape   = [input_batch, output_channel, 1, 1]

    ##########################创建input/output################################
    model_input_name  = "input0"
    model_output_name = "output0"

    input = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            input_shape)

    output = onnx.helper.make_tensor_value_info(
            model_output_name, 
            onnx.TensorProto.FLOAT, 
            output_shape)
    

    ##########################创建第一个conv节点##############################
    conv1_output_name = "conv2d_1.output"
    conv1_in_ch       = input_channel
    conv1_out_ch      = 32
    conv1_kernel      = 3
    conv1_pads        = 1

    # 创建conv节点的权重信息
    conv1_weight    = np.random.rand(conv1_out_ch, conv1_in_ch, conv1_kernel, conv1_kernel)
    conv1_bias      = np.random.rand(conv1_out_ch)

    conv1_weight_name = "conv2d_1.weight"
    conv1_weight_initializer = create_initializer_tensor(
        name         = conv1_weight_name,
        tensor_array = conv1_weight,
        data_type    = onnx.TensorProto.FLOAT)


    conv1_bias_name  = "conv2d_1.bias"
    conv1_bias_initializer = create_initializer_tensor(
        name         = conv1_bias_name,
        tensor_array = conv1_bias,
        data_type    = onnx.TensorProto.FLOAT)

    # 创建conv节点，注意conv节点的输入有3个: input, w, b
    conv1_node = onnx.helper.make_node(
        name         = "conv2d_1",
        op_type      = "Conv",
        inputs       = [
            model_input_name, 
            conv1_weight_name,
            conv1_bias_name
        ],
        outputs      = [conv1_output_name],
        kernel_shape = [conv1_kernel, conv1_kernel],
        pads         = [conv1_pads, conv1_pads, conv1_pads, conv1_pads],
    )

    ##########################创建一个BatchNorm节点###########################
    bn1_output_name = "batchNorm1.output"

    # 为BN节点添加权重信息
    bn1_scale = np.random.rand(conv1_out_ch)
    bn1_bias  = np.random.rand(conv1_out_ch)
    bn1_mean  = np.random.rand(conv1_out_ch)
    bn1_var   = np.random.rand(conv1_out_ch)

    # 通过create_initializer_tensor创建权重，方法和创建conv节点一样
    bn1_scale_name = "batchNorm1.scale"
    bn1_bias_name  = "batchNorm1.bias"
    bn1_mean_name  = "batchNorm1.mean"
    bn1_var_name   = "batchNorm1.var"

    bn1_scale_initializer = create_initializer_tensor(
        name         = bn1_scale_name,
        tensor_array = bn1_scale,
        data_type    = onnx.TensorProto.FLOAT)
    bn1_bias_initializer = create_initializer_tensor(
        name         = bn1_bias_name,
        tensor_array = bn1_bias,
        data_type    = onnx.TensorProto.FLOAT)
    bn1_mean_initializer = create_initializer_tensor(
        name         = bn1_mean_name,
        tensor_array = bn1_mean,
        data_type    = onnx.TensorProto.FLOAT)
    bn1_var_initializer  = create_initializer_tensor(
        name         = bn1_var_name,
        tensor_array = bn1_var,
        data_type    = onnx.TensorProto.FLOAT)

    # 创建BN节点，注意BN节点的输入信息有5个: input, scale, bias, mean, var
    bn1_node = onnx.helper.make_node(
        name    = "batchNorm1",
        op_type = "BatchNormalization",
        inputs  = [
            conv1_output_name,
            bn1_scale_name,
            bn1_bias_name,
            bn1_mean_name,
            bn1_var_name
        ],
        outputs=[bn1_output_name],
    )

    ##########################创建一个ReLU节点###########################
    relu1_output_name = "relu1.output"

    # 创建ReLU节点，ReLU不需要权重，所以直接make_node就好了
    relu1_node = onnx.helper.make_node(
        name    = "relu1",
        op_type = "Relu",
        inputs  = [bn1_output_name],
        outputs = [relu1_output_name],
    )

    ##########################创建一个AveragePool节点####################
    avg_pool1_output_name = "avg_pool1.output"

    # 创建AvgPool节点，AvgPool不需要权重，所以直接make_node就好了
    avg_pool1_node = onnx.helper.make_node(
        name    = "avg_pool1",
        op_type = "GlobalAveragePool",
        inputs  = [relu1_output_name],
        outputs = [avg_pool1_output_name],
    )

    ##########################创建第二个conv节点##############################

    # 创建conv节点的属性
    conv2_in_ch  = conv1_out_ch
    conv2_out_ch = output_channel
    conv2_kernel = 1
    conv2_pads   = 0

    # 创建conv节点的权重信息
    conv2_weight    = np.random.rand(conv2_out_ch, conv2_in_ch, conv2_kernel, conv2_kernel)
    conv2_bias      = np.random.rand(conv2_out_ch)
    
    conv2_weight_name = "conv2d_2.weight"
    conv2_weight_initializer = create_initializer_tensor(
        name         = conv2_weight_name,
        tensor_array = conv2_weight,
        data_type    = onnx.TensorProto.FLOAT)

    conv2_bias_name  = "conv2d_2.bias"
    conv2_bias_initializer = create_initializer_tensor(
        name         = conv2_bias_name,
        tensor_array = conv2_bias,
        data_type    = onnx.TensorProto.FLOAT)

    # 创建conv节点，注意conv节点的输入有3个: input, w, b
    conv2_node = onnx.helper.make_node(
        name         = "conv2d_2",
        op_type      = "Conv",
        inputs       = [
            avg_pool1_output_name,
            conv2_weight_name,
            conv2_bias_name
        ],
        outputs      = [model_output_name],
        kernel_shape = [conv2_kernel, conv2_kernel],
        pads         = [conv2_pads, conv2_pads, conv2_pads, conv2_pads],
    )

    ##########################创建graph##############################
    graph = onnx.helper.make_graph(
        name    = "sample-convnet",
        inputs  = [input],
        outputs = [output],
        nodes   = [
            conv1_node, 
            bn1_node, 
            relu1_node, 
            avg_pool1_node, 
            conv2_node],
        initializer =[
            conv1_weight_initializer, 
            conv1_bias_initializer,
            bn1_scale_initializer, 
            bn1_bias_initializer,
            bn1_mean_initializer, 
            bn1_var_initializer,
            conv2_weight_initializer, 
            conv2_bias_initializer
        ],
    )

    ##########################创建model##############################
    model = onnx.helper.make_model(graph, producer_name="onnx-sample")
    model.opset_import[0].version = 12
    
    ##########################验证&保存model##############################
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    print("Congratulations!! Succeed in creating {}.onnx".format(graph.name))
    onnx.save(model, "../models/sample-convnet.onnx")


# 使用onnx.helper创建一个最基本的ConvNet
#         input (ch=3, h=64, w=64)
#           |
#          Conv (in_ch=3, out_ch=32, kernel=3, pads=1)
#           |
#        BatchNorm
#           |
#          ReLU
#           |
#         AvgPool
#           |
#          Conv (in_ch=32, out_ch=10, kernel=1, pads=0)
#           |
#         output (ch=10, h=1, w=1)

if __name__ == "__main__":
    main()

