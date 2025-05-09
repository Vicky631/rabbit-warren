#!/bin/bash

# 设置要监控的 GPU 编号（从 0 开始）
GPU_INDEX=0

# 设置显存使用阈值（百分比）
THRESHOLD=99

# 要执行的命令列表
COMMANDS=("python test1.py" "python test2.py")

# 获取显存使用情况的函数
get_gpu_memory_usage() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed 's/ //g' | awk "NR==$((GPU_INDEX+1))"
}

# 获取 GPU 总显存
get_total_gpu_memory() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed 's/ //g' | awk "NR==$((GPU_INDEX+1))"
}

# 监控显存使用情况
while true; do
  USED_MEMORY=$(get_gpu_memory_usage)
  TOTAL_MEMORY=$(get_total_gpu_memory)
  
  # 如果获取的值为空，说明 `nvidia-smi` 可能无法正确执行
  if [[ -z "$USED_MEMORY" || -z "$TOTAL_MEMORY" ]]; then
    echo "无法获取 GPU 显存信息，请检查 nvidia-smi 命令是否可用！"
    exit 1
  fi

  # 计算显存使用率
  if [[ "$TOTAL_MEMORY" -gt 0 ]]; then
      USAGE=$(awk "BEGIN {printf \"%.2f\", ($USED_MEMORY / $TOTAL_MEMORY) * 100}")
  else
      USAGE=100  # 避免错误情况
  fi

  # 检查显存使用率是否低于阈值
  if (( $(awk "BEGIN {print ($USAGE < $THRESHOLD) ? 1 : 0}") )); then
    echo "显存使用率为 $USAGE%，低于 $THRESHOLD%，开始执行命令..."

    for CMD in "${COMMANDS[@]}"; do
      echo "执行命令: $CMD"
      bash -c "$CMD"

      # 如果命令执行失败，则停止执行后续命令
      if [ $? -ne 0 ]; then
        echo "命令 $CMD 执行失败，停止执行后续命令。"
        break
      fi
    done

    # 执行完命令后可以选择退出循环
    break
  else
    echo "显存使用率为 $USAGE%，等待下一次检查..."
  fi
  
  # 设置检查间隔（秒）
  sleep 10
done
