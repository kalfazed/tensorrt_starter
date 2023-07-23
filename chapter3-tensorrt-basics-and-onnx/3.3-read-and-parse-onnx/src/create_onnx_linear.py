import onnx
from onnx import helper
from onnx import TensorProto

# 理解onnx中的组织结构
#   - ModelProto (描述的是整个模型的信息)
#   --- GraphProto (描述的是整个网络的信息)
#   ------ NodeProto (描述的是各个计算节点，比如conv, linear)
#   ------ TensorProto (描述的是tensor的信息，主要包括权重)
#   ------ ValueInfoProto (描述的是input/output信息)
#   ------ AttributeProto (描述的是node节点的各种属性信息)


def create_onnx():
    # 创建ValueProto
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 10])

    # 创建NodeProto
    mul = helper.make_node('Mul', ['a', 'x'], 'c', "multiply")
    add = helper.make_node('Add', ['c', 'b'], 'y', "add")

    # 构建GraphProto
    graph = helper.make_graph([mul, add], 'sample-linear', [a, x, b], [y])

    # 构建ModelProto
    model = helper.make_model(graph)

    # 检查model是否有错误
    onnx.checker.check_model(model)
    # print(model)

    # 保存model
    onnx.save(model, "../models/sample-linear.onnx")

    return model


if __name__ == "__main__":
    model = create_onnx()
