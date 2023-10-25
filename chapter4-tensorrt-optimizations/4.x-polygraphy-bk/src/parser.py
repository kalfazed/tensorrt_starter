import onnx
import numpy as np

# 注意，因为weight是以字节的形式存储的，所以要想读，需要转变为float类型
def read_weight(initializer: onnx.TensorProto):
    shape = initializer.dims
    data  = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(shape)
    print("\n**************parse weight data******************")
    print("initializer info: \
            \n\tname:      {} \
            \n\tdata:    \n{}".format(initializer.name, data))
    

def parse_onnx(model: onnx.ModelProto):
    graph        = model.graph
    initializers = graph.initializer
    nodes        = graph.node
    inputs       = graph.input
    outputs      = graph.output

    print("\n**************parse input/output*****************")
    for input in inputs:
        input_shape = []
        for d in input.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                input_shape.append(None)
            else:
                input_shape.append(d.dim_value)
        print("Input info: \
                \n\tname:      {} \
                \n\tdata Type: {} \
                \n\tshape:     {}".format(input.name, input.type.tensor_type.elem_type, input_shape))

    for output in outputs:
        output_shape = []
        for d in output.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                output_shape.append(None)
            else:
                output_shape.append(d.dim_value)
        print("Output info: \
                \n\tname:      {} \
                \n\tdata Type: {} \
                \n\tshape:     {}".format(input.name, output.type.tensor_type.elem_type, input_shape))

    print("\n**************parse node************************")
    for node in nodes:
        print("node info: \
                \n\tname:      {} \
                \n\top_type:   {} \
                \n\tinputs:    {} \
                \n\toutputs:   {}".format(node.name, node.op_type, node.input, node.output))

    print("\n**************parse initializer*****************")
    for initializer in initializers:
        print("initializer info: \
                \n\tname:      {} \
                \n\tdata_type: {} \
                \n\tshape:     {}".format(initializer.name, initializer.data_type, initializer.dims))

