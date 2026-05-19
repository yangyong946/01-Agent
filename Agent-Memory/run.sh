#!/bin/bash
# run.sh - 一键部署并运行Demo
echo "===== 环境配置 ====="
pip install -r requirements.txt
echo "===== 运行Demo ====="
python demo.py
echo "===== 运行对比实验 ====="
python experiment.py
echo "===== 实验结果已保存至 experiment_results.csv ====="