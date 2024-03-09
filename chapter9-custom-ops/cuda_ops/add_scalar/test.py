import torch
import logging
from cuda_ops.add_scalar import add_scalar
from cuda_ops.utils.logger import Logger


def main():
    logger = Logger()
    logger.setLevel(logging.DEBUG)

    tensor = torch.randn(6, 224, 224, 3, device='cuda')
    scalar = 0.5
    logger.printTensorInformation(tensor, "before")

    # CPU version
    result = tensor.cpu() + scalar
    logger.printTensorInformation(result, "CPU")

    # GPU version
    add_scalar.add_scalar_d(tensor, scalar)  # apply add scalar in cuda
    logger.printTensorInformation(tensor, "GPU")

    # Compare
    logger.check(tensor, result)


if __name__ == '__main__':
    main()
