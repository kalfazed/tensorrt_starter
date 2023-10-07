import ctypes
import os 
import numpy as np
import tensorrt as trt
import logging

from trt_model import test_logger, console_handler, file_handler
from trt_model import build_network, inference, validation

def CustomSgemmCPU(inputH, weight):
    return np.matmul(inputH[0], weight)

def getCustomSgemmPlugin(weight) -> trt.tensorrt.IPluginV2:
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "customSgemm":
            parameterList = []
            parameterList.append(trt.PluginField("weight", np.float32(weight), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("k", np.int32(weight.shape[0]), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("n", np.int32(weight.shape[1]), trt.PluginFieldType.INT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None


def customSgemmTest(input, weight):
    current_path = os.path.dirname(__file__)
    soFile       = current_path + "/../../lib/custom-plugin.so"

    ctypes.cdll.LoadLibrary(soFile)
    plugin       = getCustomSgemmPlugin(weight)
    name         = plugin.plugin_type
    trtFile      = current_path + "/../../models/engine/%s-Dim%s.engine" % (name, str(len(shape)))
    testCase     = "<input_shape=%s,weight_shape=%s>" % (input.shape, weight.shape)
    test_logger.info("Test '%s':%s" % (name, testCase))

    # sgemm input shape:  [B, M, K]
    # sgemm weight shape: [K, N]
    k, n         = weight.shape[0], weight.shape[1]
    b, m         = input.shape[0], input.shape[1]
    shape        = [b, m, k]
    shape_opt    = {
            "min": [1, 1, k],
            "opt": [b, m, k],
            "max": [b * 2, m * 2, k]}

    #################################################################
    ################### 从这里开始是builder的部分 ######################
    #################################################################
    engine = build_network(trtFile, shape_opt, shape, plugin)
    if (engine == None):
        exit()
    
    exit()
    #################################################################
    ################### 从这里开始是infer的部分 ########################
    #################################################################
    nInput, nIO, bufferH = inference(engine, shape)

    #################################################################
    ################# 从这里开始是validation的部分 #####################
    #################################################################
    outputCPU = CustomSgemmCPU(bufferH[:nInput], weight)
    res       = validation(nInput, nIO, bufferH, outputCPU)

    if (res):
        test_logger.info("Test '%s':%s finish!\n" % (plugin.plugin_type, testCase))
    else:
        test_logger.error("Test '%s':%s failed!\n" % (plugin.plugin_type, testCase))
        exit()

def unit_test():
    b      = 8
    m      = 32
    k      = 16
    n      = 16
    input  = np.random.rand(b, m, k).astype(np.float32) * 2 - 1  # input范围在(-1, 1)
    weight = np.random.rand(k, n).astype(np.float32) * 2 - 1     # weight范围在(-1, 1)
    customSgemmTest(input, weight)

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    np.random.seed(1)

    test_logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    test_logger.info("Starting unit test...")
    unit_test()
    test_logger.info("All tests are passed!!")
