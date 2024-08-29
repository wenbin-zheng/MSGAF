#!/bin/bash

set -e
cd ..

AVT="daic_audio_mfcc daic_audio_egemaps daic_facial_aus daic_gaze daic_head_pose daic_text "

python main.py --name audiovisual-run-1 --model_args.num_layers 8 --model_args.self_attn_num_heads 4 --model_args.self_attn_dim_head 64 --save_model 1 --n_temporal_windows 1 --seconds_per_window 6 --scheduler cosine --group baseline-daicwoz-modality-ablation --mode dryrun --epochs 200 --batch_size 5 --scheduler_args.max_lr 0.0001 --scheduler_args.end_epoch 200 --config_file configs/train_configs/baseline_daicwoz_config.yaml --dataset daic-woz --env reading-between-the-frames --use_modalities $AVT
