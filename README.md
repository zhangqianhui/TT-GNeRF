# TT-GNeRF
**Training and Tuning Generative Neural Radiance Fields for Attribute-Conditional 3D-Aware Face Generation**  
[Jichao Zhang](https://zhangqianhui.github.io/), [Aliaksandr Siarohin](https://scholar.google.com/citations?user=uMl5-k4AAAAJ&hl=en), [Yahui Liu](https://scholar.google.com/citations?hl=en&user=P8qd0rEAAAAJ), [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en), 
[Nicu Sebe](http://disi.unitn.it/~sebe/), [Wei Wang](https://weiwangtrento.github.io/) <br>
[paper](https://arxiv.org/pdf/2208.12550.pdf) <br>
University of Trento, Snap Research, ETH Zurich

*Attribute Editing results with multple-view generation*
![](./imgs/teaser_opti.gif)

*Geometry Editing results with multple-view generation*
![](./imgs/geometry.gif)

*GAN inversion for Real Image Editing with multiple-view generation*
![](./imgs/real.gif)

## Environments

```
conda create -n ttgnerf python=3.6
```
```
pip install -r requirement.txt
```

## Demo 


### Editing (EG3D)

```
python train_step_Triot_styleflow.py --outdir=[output_path] \
            --network=[pretrained eg3d model] \
            --dataset_path [our dataset path] \
            --csvpath [label path] \
            --cnf_path=[cnf pretrained model path] \
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
```

### Inversion (EG3D)

PTI method
```
python inversionv2.py --outdir=[output_path_v1] --trunc=0.7 --shapes=true --seeds=0-3 \
    --network=[pretrained eg3d model] --reload_modules True --file_id 20 &

wait

python3 inversion_PTI.py --outdir=[output_path_v2] \
    --trunc=0.7 --shapes=true --seeds=0-3 \
    --network=[output_path_v1] --reload_modules True --file_id 20

```

TRIOT
```
python train_step_Triot_PTI_styleflow.py --outdir=[output_path_v3] \
            --latent_dir [output_path_v1] \
            --network=[output_path_v2] \
            --cnf_path=[cnf pretrained model path] \
            --dataset_path=[our dataset path] \
            --csvpath [label path] \
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
```


### Reference Image Geometry Transfer

```
python train_step_reference_geometry_editing.py --outdir=[output_path] --batch=1 \
            --gen_pose_cond=True --num_steps 400 \
            --w_dir [our dataset path] \
            --resolution 512 --truncation_psi 0.7 --id 13 --ref_id 41
```

## Pretrained Model and Dataset

[our dataset path]:
[cnf pretrained model path]:
[label path]:
[pretrained eg3d model]: 

To-Do List (Next Week)

# Questions

If you have any questions/comments, feel free to open a github issue or pull a request or e-mail to the author Jichao Zhang (jichao.zhang@unitn.it).

# Reference code

We would like to thank [EG3D](https://github.com/NVlabs/eg3d) for providing such a great and powerful codebase.