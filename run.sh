#13.复现
#====================================================================
# python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --batch_size=8 \
#     --output_dir checkpoint \




# python main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --eval \
#     --batch_size 1 \
#     --resume checkpoints/checkpoint/checkpoint0149.pth
#====================================================================


#====================================================================
torchrun \
    --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    cmh_reltr_main.py \
        --dataset vg \
        --img_folder /root/autodl-tmp/vg/images/ \
        --ann_path /root/autodl-tmp/vg/ \
        --batch_size=4 \
        --output_dir "./checkpoint_WDB89" \
#====================================================================



