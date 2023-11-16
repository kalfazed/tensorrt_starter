import onnx_graphsurgeon as gs
import numpy as np
import onnx

def load_model(model : onnx.ModelProto):
    graph = gs.import_onnx(model)
    print(graph.inputs)
    print(graph.outputs)

def main() -> None:
    model = onnx.load("../models/example_two_head.onnx")

    graph = gs.import_onnx(model)
    tensors = graph.tensors()

    print(tensors)
    print(tensors["input0"])
    print(tensors["output0"])
    print(tensors["output1"])
    print(tensors["onnx::MatMul_8"])
    graph.inputs = [
            tensors["input0"].to_variable(dtype=np.float32, shape=(1, 1, 1, 4))]
    graph.outputs = [
            tensors["output0"].to_variable(dtype=np.float32, shape=(1, 1, 1, 3))]
            # tensors["onnx::MatMul_8"].to_variable(dtype=np.float32, shape=(1, 1, 1, 3))]
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), "../models/example_two_head_removed.onnx")


if __name__ == "__main__":
    main()

