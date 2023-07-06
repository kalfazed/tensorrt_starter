#!/bin/bash
# how to use:
# ./infer.sh ${input.trt} ${tag}

IFS=. file=(${1})
PREFIX=${file[0]}

TAG=""
MODE=build
ROOT=log

DIR=${ROOT}/${PREFIX}/${MODE}

if [[ ${2} != "" ]]
then
        TAG=${2}_
fi

mkdir -p ${ROOT}
mkdir -p ${ROOT}/${PREFIX}
mkdir -p ${ROOT}/${PREFIX}/${MODE}

trtexec --onnx=${PREFIX}.onnx \
        --memPoolSize=workspace:2048 \
        --saveEngine=${PREFIX}.engine \
        --verbose \
        --profilingVerbosity=layer_names_only \
        --dumpOutput \
        --dumpProfile \
        --dumpLayerInfo \
        --exportOutput=${DIR}/build_output.log\
        --exportProfile=${DIR}/build_profile.log \
        --exportLayerInfo=${DIR}/build_layer_info.log \
        --warmUp=200 \
        --iterations=50 \
        > ${DIR}/build.log