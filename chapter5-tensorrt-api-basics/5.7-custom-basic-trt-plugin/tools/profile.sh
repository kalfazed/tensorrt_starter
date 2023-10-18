#!/bin/bash
# how to use:
# ./profile.sh ${input.engine} ${tag}

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

nsys profile \
        --output=${DIR}/${PREFIX} \
        --force-overwrite true \
        trtexec --loadEngine=${PREFIX}.engine \
                --warmUp=200 \
                --iterations=50 \