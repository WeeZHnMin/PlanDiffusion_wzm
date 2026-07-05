#!/usr/bin/env bash
# 等待指定数量的 GPU 空闲后立即启动训练
# 用法: bash wait_and_train.sh [需要的卡数=2] [显存阈值MB=2000] [轮询间隔秒=30]

NEED_GPUS=${1:-2}
MEM_THRESHOLD=${2:-2000}  # 已用显存低于此值(MB)视为空闲
POLL_SEC=${3:-30}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date '+%H:%M:%S')] 等待 ${NEED_GPUS} 张空闲卡（已用显存 < ${MEM_THRESHOLD} MB）..."

while true; do
    # 收集所有显存占用量（MiB），一行一张卡
    mapfile -t MEM_USED < <(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
    )

    TOTAL_GPUS=${#MEM_USED[@]}
    if [[ $TOTAL_GPUS -eq 0 ]]; then
        echo "[$(date '+%H:%M:%S')] nvidia-smi 无输出，重试..." ; sleep "$POLL_SEC" ; continue
    fi

    # 找出空闲卡的索引
    FREE_IDS=()
    for i in "${!MEM_USED[@]}"; do
        used="${MEM_USED[$i]// /}"
        if [[ $used -lt $MEM_THRESHOLD ]]; then
            FREE_IDS+=("$i")
        fi
    done

    echo "[$(date '+%H:%M:%S')] 空闲卡: [${FREE_IDS[*]}]  (${#FREE_IDS[@]}/${TOTAL_GPUS})"

    if [[ ${#FREE_IDS[@]} -ge $NEED_GPUS ]]; then
        # 取前 NEED_GPUS 张
        SELECTED=("${FREE_IDS[@]:0:$NEED_GPUS}")
        GPU_STR=$(IFS=,; echo "${SELECTED[*]}")
        echo "[$(date '+%H:%M:%S')] 抢到卡 [${GPU_STR}]，启动训练！"
        break
    fi

    sleep "$POLL_SEC"
done

export CUDA_VISIBLE_DEVICES="$GPU_STR"
echo "[$(date '+%H:%M:%S')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nproc_per_node="$NEED_GPUS" -m node_diffusion_room_tri.train \
    --data_path  data/processed/node_diffusion_room_tri/graph_dataset.npz \
    --save_dir   checkpoints/node_diffusion_room_tri \
    --bert       models/bert-base-uncased \
    --val_jsonl  data/jsonl/val_graph_dataset_18k5.jsonl \
    --resume     checkpoints/node_diffusion_room_tri/latest.pt \
    --batch_size 512 \
    --total_steps 350000 \
    --val_interval 5000 \
    --val_n      224 \
    --ddim_steps 200 \
    --val_batch  16
