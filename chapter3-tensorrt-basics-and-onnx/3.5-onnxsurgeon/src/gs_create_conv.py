import onnx_graphsurgeon as gs
import numpy as np
import onnx

# onnx_graph_surgeon(gs)中的IR会有以下三种结构
# Tensor
#    -- 有两种类型
#       -- Variable:  主要就是那些不到推理不知道的变量
#       -- Constant:  不用推理时，而在推理前就知道的变量
# Node
#    -- 跟onnx中的NodeProto差不多
# Graph
#    -- 跟onnx中的GraphProto差不多

def main() -> None:
    input = gs.Variable(
            name  = "input0",
            dtype = np.float32,
            shape = (1, 3, 224, 224))

    weight = gs.Constant(
            name  = "conv1.weight",
            values = np.random.randn(5, 3, 3, 3))

    bias   = gs.Constant(
            name  = "conv1.bias",
            values = np.random.randn(5))
    
    output = gs.Variable(
            name  = "output0",
            dtype = np.float32,
            shape = (1, 5, 224, 224))

    node = gs.Node(
            op      = "Conv",
            inputs  = [input, weight, bias],
            outputs = [output],
            attrs   = {"pads":[1, 1, 1, 1]})

    graph = gs.Graph(
            nodes   = [node],
            inputs  = [input],
            outputs = [output])

    model = gs.export_onnx(graph)

    onnx.save(model, "../models/sample-conv.onnx")



# 使用onnx.helper创建一个最基本的ConvNet
#         input (ch=3, h=64, w=64)
#           |
#          Conv (in_ch=3, out_ch=32, kernel=3, pads=1)
#           |
#         output (ch=5, h=64, w=64)

if __name__ == "__main__":
    main()

