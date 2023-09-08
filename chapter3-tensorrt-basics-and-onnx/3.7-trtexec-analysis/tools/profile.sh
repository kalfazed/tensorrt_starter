#!/bin/bash
# how to use:
#   bash tools/profile.sh ${input.engine} 

IFS=. file=(${1})
IFS=/ file=(${file})
IFS=
PREFIX=${file[2]}


if [[ ${2} != "" ]]
then
        PREFIX=${PREFIX}-${2}
fi

MODE="profile"
ONNX_PATH="models"
BUILD_PATH="build"
ENGINE_PATH=$BUILD_PATH/engines
LOG_PATH=${BUILD_PATH}"/log/"${PREFIX}"/"${MODE}

mkdir -p ${ENGINE_PATH}
mkdir -p $LOG_PATH

nsys profile \
        --output=${LOG_PATH}/${PREFIX} \
        --force-overwrite true \
        trtexec --loadEngine=${ENGINE_PATH}/${PREFIX}.engine \
                --warmUp=0 \
                --duration=0 \
                --iterations=20 \
                --noDataTransfers \
    > ${LOG_PATH}/profile.log

