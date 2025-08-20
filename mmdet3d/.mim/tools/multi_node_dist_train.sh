#!/usr/bin/env bash

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
--nproc_per_node 2 --master_addr "localhost" \
--node_rank 0 --master_port "12364" --nnodes 1 \
$(dirname "$0")/train.py $CONFIG --launcher pytorch --resume-from /root/gekim/distill-bev/outputs/centerpoint_to_bevdepth4d_r50_virtual/best3+t0.5/epoch_8.pth ${@:2}