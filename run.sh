# python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --batch_size=8 \
#     --output_dir checkpoint \


python -m torch.distributed.launch  \
    --nproc_per_node=2 \
    --use_env main.py \
    --dataset vg \
    --img_folder data/vg/images/ \
    --ann_path data/vg/ \
    --batch_size=8 \
    --output_dir checkpoint2 \