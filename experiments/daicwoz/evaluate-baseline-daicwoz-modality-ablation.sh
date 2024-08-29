#!/bin/bash

set -e
cd ..

python evaluate.py
--eval_config
configs/eval_configs/eval_daicwoz_test_config.yaml
--output_dir
baseline-daicwoz-modality-ablation
--checkpoint_kind
best
--name
audiovisual-run-5
--n_temporal_windows
1
--seconds_per_window
6
--batch_size
5
--group
baseline-daicwoz-modality-ablation
--env
reading-between-the-frames