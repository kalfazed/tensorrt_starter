#!/bin/bash
# how to use:
# ./infer.sh ${input.trt} ${tag}

IFS=. file=(${1})
PREFIX=${file[0]}

TAG=""
MODE=infer
ROOT=log

DIR=${ROOT}/${PREFIX}/${MODE}

if [[ ${2} != "" ]]
then
        TAG=${2}_
fi

mkdir -p ${ROOT}
mkdir -p ${ROOT}/${PREFIX}
mkdir -p ${ROOT}/${PREFIX}/${MODE}

trtexec --loadEngine=${PREFIX}.engine \
        --verbose \
        --dumpOutput \
        --dumpProfile \
        --dumpLayerInfo \
        --exportOutput=${DIR}/infer_output.log\
        --exportProfile=${DIR}/infer_profile.log \
        --exportLayerInfo=${DIR}/infer_layer_info.log \
        --warmUp=200 \
        --iterations=50 \
        > ${DIR}/infer.log