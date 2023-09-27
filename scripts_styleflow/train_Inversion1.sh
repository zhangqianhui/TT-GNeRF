#!/bin/bash

python inversionv2.py --outdir=/nfs/data_chaos/jzhang/results/tvcg/attribute_editing_triot_8_13/ --trunc=0.7 --shapes=true --seeds=0-3 \
    --network=/nfs/data_chaos/jzhang/dataset/pretrained/eg3d/model_2000_G.pkl --reload_modules True --file_id 20 &

wait

python3 inversion_PTI.py --outdir=/nfs/data_chaos/jzhang/results/tvcg/attribute_editing_triot_8_13/inversion_ptiv2 \
    --trunc=0.7 --shapes=true --seeds=0-3 \
    --network=/nfs/data_chaos/jzhang/results/tvcg/attribute_editing_triot_8_13/ --reload_modules True --file_id 20

