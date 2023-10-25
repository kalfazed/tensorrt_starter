import onnx

def main(): 

    model = onnx.load("../models/sample-linear.onnx")
    onnx.checker.check_model(model)

    graph        = model.graph
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


if __name__ == "__main__":
    main()

