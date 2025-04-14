#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=6 train.py
