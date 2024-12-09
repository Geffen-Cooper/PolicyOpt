#!/bin/bash

python preprocess.py \
    --root_dir ~/Projects/data/dsads \
    --body_part right_leg \
    --activity_list 9 11 15 17 18  \
    # --activity_list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
    # 0: --- sitting
    # 1: --- standing
    # 2: --- lying on back
    # 3: --- lying on right side
    # 4: +++ ascending stairs
    # 5: +++ descending stairs
    # 6: --- standing in elevator
    # 7: +--  moving in elevator
    # 8: +++ walking in parking lot
    # 9: +++ walking on flat treadmill
    # 10: +++ walking on inclined treadmill
    # 11: +++ running on treadmill
    # 12: +++ exercising on stepper
    # 13: +++ exercising on cross trainer
    # 14: +++ cycling on exercise bike horizontal
    # 15: +++ cycling on exercise bike vertical
    # 16: +-- rowing
    # 17: +++ jumping
    # 18: +++ playing basketball
