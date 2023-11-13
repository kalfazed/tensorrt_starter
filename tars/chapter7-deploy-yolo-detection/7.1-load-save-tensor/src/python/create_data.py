import os
import numpy as np
from logger import init_logger

logger = init_logger()

def createData(data_path, data_label, data_shape):
    dataDict = {}
    dataDict[data_label] = np.random.rand(*data_shape).astype(np.float32)
    np.savez(data_path, **dataDict)
    logger.info("Succeeded saving data as .npz file!")
    logger.info("Tensor shape:{}".format(dataDict[data_label].shape))
    logger.info("Tensor values:")
    print(dataDict[data_label])
    return

def loadData(data_path, data_label):
    dataDict = np.load(data_path)
    logger.info("Succeeded loaded data as .npz file!")
    logger.info("Tensor shape:{}".format(dataDict[data_label].shape))
    logger.info("Tensor values:")
    print(dataDict[data_label])
    return

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8f}'.format})
    current_path = os.path.dirname(__file__)
    data_path    = current_path + "/../../data/data_python.npz"
    data_label   = "data_python"
    data_shape   = (2, 3, 4, 4)

    # createData(data_path, data_label, data_shape)

    data_path    = current_path + "/../../data/data_cpp.npz"
    data_label   = "data_cpp"
    loadData(data_path, data_label)
