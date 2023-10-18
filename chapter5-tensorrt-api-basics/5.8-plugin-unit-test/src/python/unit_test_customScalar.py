import ctypes
import os 
import numpy as np
import tensorrt as trt
import logging

from trt_model import test_logger, console_handler, file_handler
from trt_model import build_network, inference, validation


def CustomScalarCPU(inputH, scalar, scale):
    return [(inputH[0] + scalar) * scale]

def getCustomScalarPlugin(scalar, scale) -> trt.tensorrt.IPluginV2:
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "customScalar":
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("scale", np.float32(scale), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None


def customScalarTest(shape, scalar, scale):
    current_path = os.path.dirname(__file__)
    soFile       = current_path + "/../../lib/custom-plugin.so"
    trtFile      = current_path + "/../../models/engine/model-Dim%s.engine" % str(len(shape))
    testCase     = "<shape=%s,scalar=%f,scale=%f>" % (shape, scalar, scale)

    ctypes.cdll.LoadLibrary(soFile)
    plugin = getCustomScalarPlugin(scalar, scale)
    test_logger.info("Test '%s':%s" % (plugin.plugin_type, testCase))

    #################################################################
    ################### 从这里开始是builder的部分 ######################
    #################################################################
    engine = build_network(trtFile, shape, plugin)
    if (engine == None):
        exit()

    #################################################################
    ################### 从这里开始是infer的部分 ########################
    #################################################################
    nInput, nIO, bufferH = inference(engine, shape)

    #################################################################
    ################# 从这里开始是validation的部分 #####################
    #################################################################
    outputCPU = CustomScalarCPU(bufferH[:nInput], scalar, scale)
    res = validation(nInput, nIO, bufferH, outputCPU)

    if (res):
        test_logger.info("Test '%s':%s finish!\n" % (plugin.plugin_type, testCase))
    else:
        test_logger.error("Test '%s':%s failed!\n" % (plugin.plugin_type, testCase))
        exit()

def unit_test():
    customScalarTest([32], 1, 10)
    customScalarTest([32, 32], 2, 5)
    customScalarTest([16, 16, 16], 1, 3)
    customScalarTest([8, 8, 8, 8], 1, 5)

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    np.random.seed(1)

    test_logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    test_logger.info("Starting unit test...")
    unit_test()
    test_logger.info("All tests are passed!!")
