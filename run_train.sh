#!/bin/bash

# 配置变量
DATASET="RML201610a"          # 数据集名称
ALGORITHM="FT"             # 算法名称
PARAM="None"              # 参数


# 目标文件夹路径（包含数据集名）
TARGET_DIR="result/${DATASET}"

# 如果目标文件夹不存在，则创建
mkdir -p "$TARGET_DIR"

# 构建日志文件名（包含所有参数）
LOG_NAME="${DATASET}_${ALGORITHM}_${PARAM}.log"
LOG_PATH="${TARGET_DIR}/${LOG_NAME}"

# 运行训练脚本并将输出保存到日志文件
python train.py \
    --dataset "$DATASET" \
    --lr 1e-3 \
    --batch_size 400 \
    --patience 50 \
    | tee "$LOG_PATH"

echo "实验日志已保存到: $LOG_PATH"