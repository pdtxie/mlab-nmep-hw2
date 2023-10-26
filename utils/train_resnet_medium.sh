#! /bin/sh

CUDA_VISIBLE_DEVICES=1,2,4,6 python ../main.py --cfg=../configs/resnet18_medium_imagenet.yaml
