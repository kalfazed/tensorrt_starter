#!/bin/sh/

docker run -it \
	--name trt-hondanet-${1} \
	-e http_proxy=172.17.0.1:5865 \
	-e https_proxy=172.17.0.1:5865 \
	-v /mnt/snd5510:/mnt/snd5510 \
	-v /mnt/snd5420:/mnt/snd5420 \
	-v /mnt/AI_Sol:/mnt/AI_Sol \
	-v /mnt/egl690217:/mnt/egl690217 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=:1 \
	172.23.17.149:4567/visionai/trt-hondanet:cuda11.4-cudnn8-tensorrt8.2-${1}
	
	
	
