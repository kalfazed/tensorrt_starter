#!/bin/sh/

docker run -it \
	--name trt_starter_${1} \
  --gpus all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 8090:22 \
	-e DISPLAY=:1 \
	trt_starter:cuda11.4-cudnn8-tensorrt8.2_${1} \
  fish

	
	
	
