#!/bin/bash

# 目标文件夹路径
TARGET_DIR="result/"

# 如果目标文件夹不存在，则创建
mkdir -p "$TARGET_DIR"

# 运行训练脚本并将输出保存到日志文件
python train.py --dataset 10a --lr 1e-3 --batch_size 400 | tee "$TARGET_DIR/RML201610a_MKAFT_dylr1e-3_bs400_crossentropy_updatelr.log"