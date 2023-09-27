#!/bin/bash

python train_step_Triot_PTI_styleflow.py --outdir=/nfs/data_chaos/jzhang/results/tvcg/attribute_editing_triot_8_13/attribute_editing_triot_pti \
            --latent_dir /nfs/data_chaos/jzhang/results/tvcg/attribute_editing_triot_8_13/ \
            --network=/nfs/data_chaos/jzhang/results/tvcg/attribute_editing_triot_8_13/inversion_ptiv2/ \
            --cnf_path=/nfs/data_chaos/jzhang/dataset/styleflow_results \
            --dataset_path=/nfs/data_chaos/jzhang/dataset/training_data_40000 \
            --csvpath '/nfs/data_chaos/jzhang/dataset/label_eg3d_7_5_attribute.csv' \
            --batch=1 \
            --gen_pose_cond=True \
            --resolution 512 \
            --gen_pose_cond True \
            --label_dim 6 \
            --truncation_psi 0.7 \
            --scale 1.2 \
            --finetune_id 3 \
            --file_id 17 \
            --num_steps 150 \
