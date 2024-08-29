#!/bin/bash

set -e
cd ..

BATCH_SIZE=8
SECONDS_PER_WINDOW=6

GROUP=baseline-edaic-modality-ablation

EVAL_TEST_CONFIG="configs/eval_configs/eval_edaic_test_config.yaml"
EVAL_VAL_CONFIG="configs/eval_configs/eval_edaic_val_config.yaml"
ENV="reading-between-the-frames"


python evaluate.py --eval_config $EVAL_VAL_CONFIG --output_dir $GROUP --checkpoint_kind best --name audiovisual-run-1 --n_temporal_windows 1 --seconds_per_window $SECONDS_PER_WINDOW --batch_size $BATCH_SIZE --group $GROUP --env $ENV

