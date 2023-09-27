# python train_step1.py --outdir='/apdcephfs/share_1330077/chjingzhang/experiments_eg3d' --batch=1 \
#             --gen_pose_cond=True --num_steps 2001 \
#             --dataset_path '/apdcephfs/share_1330077/chjingzhang/dataset/eg3d/training_data_40000' \
#             --csvpath '/apdcephfs/share_1330077/chjingzhang/dataset/eg3d/label_eg3d_7_5_attribute.csv' \
#             --resolution 512 \
#             --gen_pose_cond True \
#             --label_dim 6 --label_w_recon 6 \
#             --label_image_recon 6 --truncation_psi 0.7 --label_cls_d 10 --is_weighted False

# python gen_video_editing.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3/out_editing_2_24 --trunc=0.7 \
#               --seeds=11 --network=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/model_1000_G.pkl --shapes False

# python train_step_reference_geometry_editing.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/geometry_editing_results --batch=1 \
#             --gen_pose_cond=True --num_steps 150 \
#             --w_dir '/apdcephfs/share_1330077/chjingzhang/experiments_eg3/out_editing_2_24' \
#             --resolution 512 --truncation_psi 0.7 --id 2 --ref_id 1
# training styleflow for eg3d

python train_flow.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/styleflow_results --batch=5 \
            --gen_pose_cond=True --num_steps 10001 \
            --dataset_path '/apdcephfs/share_1330077/chjingzhang/dataset/eg3d/training_data_40000' \
            --csvpath '/apdcephfs/share_1330077/chjingzhang/dataset/eg3d/label_eg3d_7_5_attribute.csv' \
            --resolution 512 \
            --gen_pose_cond True \
            --label_dim 6 --truncation_psi 0.7

# attribute gen_video_editing

# python train_step_Triot.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/attribute_editing_triot_3_15 \
#             --latent_dir /apdcephfs/share_1330077/chjingzhang/experiments_eg3/out_editing_intensity1.5 \
#             --network=/apdcephfs/share_1330077/chjingzhang/pretrained_model/eg3d/model_2000_G.pkl \
#             --batch=1 \
#             --gen_pose_cond=True \
#             --resolution 512 \
#             --gen_pose_cond True \
#             --label_dim 6 \
#             --truncation_psi 0.7 \
#             --scale 1.0 \
#             --finetune_id 3 \
#             --file_id 10 \
#             --num_steps 150 \

# python train_step1.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/attribute_editing_3_12_v7 --batch=6 \
#             --gen_pose_cond=True --num_steps 2001 \
#             --dataset_path '/apdcephfs/share_1330077/chjingzhang/dataset/eg3d/training_data_40000' \
#             --csvpath '/apdcephfs/share_1330077/chjingzhang/dataset/eg3d/label_eg3d_7_5_attribute.csv' \
#             --resolution 512 \
#             --gen_pose_cond True \
#             --label_dim 6 --label_w_recon 6 \
#             --label_image_recon 6 --truncation_psi 0.7 --label_cls_d 10 --is_weighted False --w_select 0

# python gen_video_editing.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3/out_editing_intensity1.5 --trunc=0.7 \
#               --seeds=11 --network=/apdcephfs/share_1330077/chjingzhang/pretrained_model/eg3d/model_2000_G.pkl --shapes False --w_select 0
# python gen_video_PTI.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/output_inversionv2 --trunc=0.7 --seeds=0-3 --grid=1x1 \
#     --network=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/output_inversionv2/model_500_PTI.pkl
#
# python train_step_Triot_PTI.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/attribute_editing_triot_pti \
#             --latent_dir /apdcephfs/share_1330077/chjingzhang/experiments_eg3d/output_inversionv2 \
#             --network=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/output_inversionv2/model_500_PTI.pkl \
#             --batch=1 \
#             --gen_pose_cond=True \
#             --resolution 512 \
#             --gen_pose_cond True \
#             --label_dim 6 \
#             --truncation_psi 0.7 \
#             --scale -1.2 \
#             --finetune_id 4 \
#             --file_id 4 \
#             --num_steps 300 \

# python inversionv2.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/output_inversionv2 --trunc=0.7 --shapes=true --seeds=0-3 \
#     --network=/apdcephfs/share_1330077/chjingzhang/pretrained_model/eg3d/model_2000_G.pkl --reload_modules True --file_id 230

# python3 inversion_PTI.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/output_inversionv2 --trunc=0.7 --shapes=true --seeds=0-3 \
#     --network=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/output_inversionv2/ --reload_modules True --file_id 230
#

# python train_step_Triot_PTI.py --outdir=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/attribute_editing_triot_pti \
#             --latent_dir /apdcephfs/share_1330077/chjingzhang/experiments_eg3d/output_inversionv2 \
#             --network=/apdcephfs/share_1330077/chjingzhang/experiments_eg3d/output_inversionv2 \
#             --batch=1 \
#             --gen_pose_cond=True \
#             --resolution 512 \
#             --gen_pose_cond True \
#             --label_dim 6 \
#             --truncation_psi 0.7 \
#             --scale -1.5 \
#             --finetune_id 5 \
#             --file_id 221 \
#             --num_steps 150