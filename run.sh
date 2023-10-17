#1.复现
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
#     --resume checkpoint/checkpoint0149.pth
#====================================================================



#====================================================================
# python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --batch_size=8 \
#     --output_dir checkpoint2 \

# export CUDA_VISIBLE_DEVICES = 0
# python main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --eval \
#     --batch_size 1 \
#     --resume checkpoint2/checkpoint0149.pth
#====================================================================



#3调试
#====================================================================
# python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --batch_size=8 \
#     --output_dir checkpoint3 \


# python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --batch_size=8 \
#     --resume '/home/minghach/Data/CMH/RelTR/checkpoint3/checkpoint0134.pth' \
#     --output_dir checkpoint3 \

# python main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --eval \
#     --batch_size 1 \
#     --resume checkpoint3/checkpoint0149.pth




#4占坑
#====================================================================
# python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --batch_size=8 \
#     --output_dir checkpoint4 \




#====================================================================