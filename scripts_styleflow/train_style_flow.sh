#!/bin/bash

python train_flow.py --outdir=./styleflow_results --batch=5 \
            --gen_pose_cond=True --num_steps 10001 \
            --dataset_path '/nfs/data_chaos/jzhang/dataset/training_data_40000' \
            --csvpath '/nfs/data_chaos/jzhang/dataset/label_eg3d_7_5_attribute.csv' \
            --network=/nfs/data_chaos/jzhang/dataset/pretrained/eg3d/model_2000_G.pkl \
            --resolution 512 \
            --gen_pose_cond True \
            --label_dim 6 --truncation_psi 0.7