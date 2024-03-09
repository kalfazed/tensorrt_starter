# setup.py

import os
import re

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(
    name,
    module,
    sources,
    sources_cuda=None,
    extra_args=None,
    extra_include_path=None
):
    sources_utils = [
        "../utils/logger.cpp",
        "../utils/timer.cpp",
        "../utils/utils.cpp"
    ]
    include_path = [
        '/home/t2-auto/packages/TensorRT-8.6.1.6/include',
    ]

    if sources_cuda is None:
        sources_cuda = []
    if extra_args is None:
        extra_args = []
    if extra_include_path is None:
        extra_include_path = []

    extra_include_path += include_path
    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
        sources += sources_utils
    else:
        raise EnvironmentError('CUDA is required to compile custom operations')

    return extension(
        name=f"{module}.{name}",
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


def _get_version(package: str) -> str:
    with open(os.path.join(package, "__version__.py")) as f:
        version = f.read()
    match = re.search("__version__ = ['\"]([^'\"]+)['\"]", version)
    assert match is not None
    return match.group(1)


setup(
    name='cuda_ops',
    version=_get_version("cuda_ops"),
    description="custom operations written in CUDA",
    author="kalfazed",
    packages=find_packages(exclude=["test"]),
    ext_modules=[
        make_cuda_ext(
            name="add_scalar",
            module="cuda_ops.add_scalar",
            sources=[
                'src/add_scalar.cpp',
                'src/add_scalar_kernel.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
