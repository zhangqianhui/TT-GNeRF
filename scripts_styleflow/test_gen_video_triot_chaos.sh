#!/bin/bash

# [10, 20, 50, 100, 150, 200, 300， 500, 1000]
python train_step_Triot_styleflow.py --outdir=/nfs/data_chaos/jzhang/results/tvcg/attribute_editing_triot_8_13 \
            --network=/nfs/data_chaos/jzhang/dataset/pretrained/eg3d/model_2000_G.pkl \
            --dataset_path '/nfs/data_chaos/jzhang/dataset/training_data_40000' \
            --csvpath '/nfs/data_chaos/jzhang/dataset/label_eg3d_7_5_attribute.csv' \
            --cnf_path=/nfs/data_chaos/jzhang/dataset/styleflow_results \
            --batch=1 \
            --gen_pose_cond=True \
            --resolution 512 \
            --label_dim 6 \
            --truncation_psi 0.7 \
            --scale 1.5 \
            --finetune_id 2 \
            --file_id 202 \
            --num_steps 100 \
            --lambda_normal 1.0