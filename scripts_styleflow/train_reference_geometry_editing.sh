#!/bin/bash

python train_step_reference_geometry_editing.py --outdir=/nfs/data_chaos/jzhang/results/tvcg/reference_editing --batch=1 \
            --gen_pose_cond=True --num_steps 400 \
            --w_dir '/nfs/data_chaos/jzhang/dataset/training_data_40000' \
            --resolution 512 --truncation_psi 0.7 --id 13 --ref_id 41