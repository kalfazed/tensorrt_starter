import onnx_graphsurgeon as gs
import numpy as np
import onnx

#####################在graph注册调用的函数########################
@gs.Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])

@gs.Graph.register()
def mul(self, a, b):
    return self.layer(op="Mul", inputs=[a, b], outputs=["mul_out_gs"])

@gs.Graph.register()
def gemm(self, a, b, trans_a=False, trans_b=False):
    attrs = {"transA": int(trans_a), "transB": int(trans_b)}
    return self.layer(op="Gemm", inputs=[a, b], outputs=["gemm_out_gs"], attrs=attrs)

@gs.Graph.register()
def relu(self, a):
    return self.layer(op="Relu", inputs=[a], outputs=["act_out_gs"])


#####################通过注册的函数进行创建网络########################
#          input (64, 64)
#            |
#           gemm (constant tensor A(64, 32))
#            |
#           add  (constant tensor B(64, 32))
#            |
#           relu
#            |
#           mul  (constant tensor C(64, 32))
#            |
#           add  (constant tensor D(64, 32))

# 初始化网络的opset
graph    = gs.Graph(opset=12)

# 初始化网络需要用的参数
consA    = gs.Constant(name="consA", values=np.random.randn(64, 32))
consB    = gs.Constant(name="consB", values=np.random.randn(64, 32))
consC    = gs.Constant(name="consC", values=np.random.randn(64, 32))
consD    = gs.Constant(name="consD", values=np.random.randn(64, 32))
input0   = gs.Variable(name="input0", dtype=np.float32, shape=(64, 64))

# 设计网络架构
gemm0    = graph.gemm(input0, consA, trans_b=True)
relu0    = graph.relu(*graph.add(*gemm0, consB))
mul0     = graph.mul(*relu0, consC)
output0  = graph.add(*mul0, consD)

# 设置网络的输入输出
graph.inputs = [input0]
graph.outputs = output0

for out in graph.outputs:
    out.dtype = np.float32

# 保存模型
onnx.save(gs.export_onnx(graph), "../models/sample-complicated-graph.onnx")


