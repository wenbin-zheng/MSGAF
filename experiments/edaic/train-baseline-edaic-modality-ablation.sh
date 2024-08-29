#!/bin/bash

set -e
cd ..

EPOCHS=200
MAX_LR=0.0001
BATCH_SIZE=8
SECONDS_PER_WINDOW=6

A="edaic_audio_mfcc edaic_audio_egemaps"
V="edaic_video_cnn_resnet edaic_video_pose_gaze_aus"
AV="edaic_audio_mfcc edaic_audio_egemaps edaic_video_pose_gaze_aus edaic_text"

WANDB_MODE=dryrun
MODEL_ARGS="--model_args.num_layers 8 --model_args.self_attn_num_heads 8 --model_args.self_attn_dim_head 32"
GROUP=baseline-edaic-modality-ablation
CONFIG_FILE="configs/train_configs/baseline_edaic_config.yaml"
ENV="reading-between-the-frames"

python main.py --run_id 1 --name audiovisual-run-1 $MODEL_ARGS --save_model 1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --scheduler cosine --group $GROUP --mode $WANDB_MODE  --epochs $EPOCHS --batch_size $BATCH_SIZE --scheduler_args.max_lr $MAX_LR --scheduler_args.end_epoch $EPOCHS --config_file $CONFIG_FILE --dataset e-daic-woz --env $ENV --use_modalities $AV
