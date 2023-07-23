import onnx_graphsurgeon as gs
import numpy as np
import onnx

def load_model(model : onnx.ModelProto):
    graph = gs.import_onnx(model)
    print(graph.inputs)
    print(graph.outputs)

def main() -> None:
    model = onnx.load("../models/swin-tiny.onnx")

    graph = gs.import_onnx(model)
    tensors = graph.tensors()

    # LayerNorm部分 
    print(tensors["374"])  # LN的input1: 1 x 3136 x 128
    print(tensors["375"])  # LN的input2: 1 x 3136 x 1
    print(tensors["383"])  # LN的输出:   1 x 3136 x 128
    graph.inputs = [
            tensors["374"].to_variable(dtype=np.float32, shape=(1, 3136, 128))]
    graph.outputs = [
            tensors["383"].to_variable(dtype=np.float32, shape=(1, 3136, 128))]
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), "../models/swin-subgraph-LN.onnx")

    # MHSA部分
    graph = gs.import_onnx(model)
    tensors = graph.tensors()
    print(tensors["457"])   # MHSA输入matmul:       64 x 49 x 128
    print(tensors["5509"])  # MHSA输入matmul的权重: 128 x 384
    print(tensors["5518"])  # MHSA输出matmul的权重: 128 x 128
    print(tensors["512"])   # MHSA输出:             64 x 49 x 128
    graph.inputs = [
            tensors["457"].to_variable(dtype=np.float32, shape=(64, 49, 128))]
    graph.outputs = [
            tensors["512"].to_variable(dtype=np.float32, shape=(64, 49, 128))]
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), "../models/swin-subgraph-MSHA.onnx")

# 我们想把swin中LayerNorm中的这一部分单独拿出来
if __name__ == "__main__":
    main()

