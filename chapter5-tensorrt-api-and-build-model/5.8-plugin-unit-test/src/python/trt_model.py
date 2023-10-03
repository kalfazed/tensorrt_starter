import os
import numpy as np
import tensorrt as trt
from cuda import cudart
from logger import set_logger

current_path = os.path.dirname(__file__)
log          = current_path + "/../../logs/custom-plugin.log"
test_logger, file_handler, console_handler = set_logger(log)

def build_network(trtFile, shape, plugin):
    # 从shared library中读取plugin
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')

    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            # 反序列化一个推理引擎
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            test_logger.error("Failed loading engine!")
            return
        test_logger.info("Succeeded loading engine!")
    else:
        # 利用python api创建一个推理引擎。这个推理引擎中只有我们准备做unit-test所需要的plugin。创建engine的流程和c++是一样的
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config  = builder.create_builder_config()

        # 为network创建一个dummy的输入，并支持动态shape
        inputT0 = network.add_input("inputT0", trt.float32, [-1 for i in shape])
        profile.set_shape(inputT0.name, [1 for i in shape], [8 for i in shape], [32 for i in shape])
        config.add_optimization_profile(profile)

        # 为network添加这个plugin所对应的layer
        pluginLayer = network.add_plugin_v2([inputT0], plugin)

        # 为network标记输出
        network.mark_output(pluginLayer.get_output(0))

        # 序列化engine并保存
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            test_logger.error("Failed building engine!")
            return
        test_logger.info("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    
    return engine


def inference(engine, shape):
    nIO         = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput      = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context     = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], shape)

    # 初始化host端的数据，根据输入的shape大小来初始化值, 同时也把存储输出的空间存储下来
    bufferH     = []
    # bufferH.append(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    bufferH.append((np.random.uniform(-1, 1, np.prod(shape)).astype(np.float32)).reshape(shape))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))

    # 初始化device端的内存，根据host端的大小来分配空间
    bufferD     = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # H2D, enqueue, D2H执行推理，并把结果返回
    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))
    context.execute_async_v3(0)
    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for b in bufferD:
        cudart.cudaFree(b)

    return nInput, nIO, bufferH

def validation(nInput, nIO, bufferH, standard):
    for i in range(nInput):
        printArrayInformation(bufferH[i], info="input[%s]" % i)
    for i in range(nInput, nIO):
        printArrayInformation(bufferH[i], info="output(plugin-impl)[%s]" % (i - nInput))
    for i in range(nInput, nIO):
        printArrayInformation(standard[i - nInput], info="output(cpu-impl)[%s]" % (i - nInput))

    # CPU端的计算，与Plugin中核函数计算结果比较
    return check(bufferH[nInput:][0], standard[0], True)


# 输出tensor的前5位和后5位数据
def printArrayInformation(x, info="", n=5):
    if 0 in x.shape:
        test_logger.debug('%s:%s' % (info, str(x.shape)))
        test_logger.debug()
        return
    test_logger.debug('%s:%s'%(info,str(x.shape)))
    test_logger.debug('\tSumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    test_logger.debug('\t%s ...  %s'%(x.reshape(-1)[:n], x.reshape(-1)[-n:]))

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    test_logger.info("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))
    return res