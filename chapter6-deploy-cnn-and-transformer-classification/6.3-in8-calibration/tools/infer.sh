#!/bin/bash
# how to use:
#   bash tools/infer.sh ${input.engine}

IFS=. file=(${1})
IFS=/ file=(${file})
IFS=
PREFIX=${file[2]}


if [[ ${2} != "" ]]
then
        PREFIX=${PREFIX}-${2}
fi

MODE="infer"
ONNX_PATH="models"
BUILD_PATH="build"
ENGINE_PATH=$BUILD_PATH/engines
LOG_PATH=${BUILD_PATH}"/log/"${PREFIX}"/"${MODE}

mkdir -p ${ENGINE_PATH}
mkdir -p $LOG_PATH

trtexec --loadEngine=${ENGINE_PATH}/${PREFIX}.engine \
        --dumpOutput \
        --dumpProfile \
        --dumpLayerInfo \
        --exportOutput=${LOG_PATH}/infer_output.log\
        --exportProfile=${LOG_PATH}/infer_profile.log \
        --exportLayerInfo=${LOG_PATH}/infer_layer_info.log \
        --warmUp=200 \
        --iterations=50 \
        > ${LOG_PATH}/infer.log
