#!/bin/bash
# how to use:
#   bash tools/build.sh ${input.onnx} ${tag}

IFS=. file=(${1})
IFS=/ file=(${file})
IFS=
PREFIX=${file[1]}


if [[ ${2} != "" ]]
then
        PREFIX=${PREFIX}-${2}
fi

MODE="build"
ONNX_PATH="models"
BUILD_PATH="build"
ENGINE_PATH=$BUILD_PATH/engines
LOG_PATH=${BUILD_PATH}"/log/"${PREFIX}"/"${MODE}

mkdir -p ${ENGINE_PATH}
mkdir -p $LOG_PATH

trtexec --onnx=${1} \
        --memPoolSize=workspace:2048 \
        --saveEngine=${ENGINE_PATH}/${PREFIX}.engine \
        --profilingVerbosity=detailed \
        --dumpOutput \
        --dumpProfile \
        --dumpLayerInfo \
        --exportOutput=${LOG_PATH}/build_output.log\
        --exportProfile=${LOG_PATH}/build_profile.log \
        --exportLayerInfo=${LOG_PATH}/build_layer_info.log \
        --warmUp=200 \
        --iterations=50 \
        --verbose \
        --fp16 \
        > ${LOG_PATH}/build.log
