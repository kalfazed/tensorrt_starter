import ctypes
import os 
import numpy as np
import tensorrt as trt
import logging

from trt_model import test_logger, console_handler, file_handler
from trt_model import build_network, inference, validation

def CustomLeakyReLUCPU(inputH, alpha):
    return [np.where(inputH[0] > 0, inputH[0], alpha * inputH[0])]

def getCustomLeakyReLUPlugin(alpha) -> trt.tensorrt.IPluginV2:
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "customLeakyReLU":
            parameterList = []
            parameterList.append(trt.PluginField("alpha", np.float32(alpha), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None


def customLeakyReLUTest(shape, alpha):
    current_path = os.path.dirname(__file__)
    soFile       = current_path + "/../../lib/custom-plugin.so"
    ctypes.cdll.LoadLibrary(soFile)

    plugin       = getCustomLeakyReLUPlugin(alpha)
    name         = plugin.plugin_type
    trtFile      = current_path + "/../../models/engine/%s-Dim%s.engine" % (name, str(len(shape)))
    testCase     = "<shape=%s,alpha=%f>" % (shape, alpha)

    test_logger.info("Test '%s':%s" % (name, testCase))

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
    outputCPU = CustomLeakyReLUCPU(bufferH[:nInput], alpha)
    res = validation(nInput, nIO, bufferH, outputCPU)

    if (res):
        test_logger.info("Test '%s':%s finish!\n" % (name, testCase))
    else:
        test_logger.error("Test '%s':%s failed!\n" % (name, testCase))
        exit()

def unit_test():
    customLeakyReLUTest([32], 0.01)
    customLeakyReLUTest([32, 32], 0.04)
    customLeakyReLUTest([16, 16, 16], 0.02)
    customLeakyReLUTest([8, 8, 8, 8], 0.1)

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    np.random.seed(1)

    test_logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    test_logger.info("Starting unit test...")
    unit_test()
    test_logger.info("All tests are passed!!")
