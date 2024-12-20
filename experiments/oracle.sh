#!/bin/bash

python test_oracle.py \
    --root_dir ~/Projects/data/dsads \
    --seed 0 \
    --duration_list 1 2 5 10 20 50
    # --min_duration 1 \
    # --max_duration 20 \