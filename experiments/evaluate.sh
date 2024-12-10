#!/bin/bash

python evaluate_opportunistic.py \
    --root_dir ~/Projects/data/dsads \
    --seed 0 \
    --duration_list 0 1 2 5 10 20 50
    # --min_duration 1 \
    # --max_duration 20 \