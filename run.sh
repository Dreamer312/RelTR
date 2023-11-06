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
#     --resume checkpoints/checkpoint/checkpoint0149.pth
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




#5  关闭了seed，使用ema
#====================================================================
# python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env cmh_main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --batch_size=8 \
#     --output_dir checkpoint5 \


# export CUDA_VISIBLE_DEVICES=0
# python main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --eval \
#     --batch_size 1 \
#     --resume checkpoints/checkpoint5/checkpoint0149.pth


# export CUDA_VISIBLE_DEVICES=1
# python cmh_main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --eval \
#     --batch_size 1 \
#     --resume checkpoints/checkpoint5/checkpoint0149.pth

#====================================================================




#7  zju调试
#====================================================================
export CUDA_VISIBLE_DEVICES=1
# python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env cmh_main.py \
#     --dataset vg \
#     --img_folder data/vg/images/ \
#     --ann_path data/vg/ \
#     --batch_size=4 \
#     --output_dir checkpoint_zju_debug \

torchrun --nproc_per_node=1 \
    --standalone \
    --nnodes=1 \
    main.py \
        --dataset vg \
        --img_folder data/vg/images/ \
        --ann_path data/vg/ \
        --batch_size=1 \
        --output_dir checkpoint_zju_debug \



#====================================================================