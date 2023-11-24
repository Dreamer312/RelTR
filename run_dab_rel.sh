#====================================================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# torchrun --nproc_per_node=8 \
#     --standalone \
#     --nnodes=1 \
#     cmh_dab_rel_main.py \
#         --dataset vg \
#         --img_folder data/vg/images/ \
#         --ann_path data/vg/ \
#         --modelname dab_detr \
#         --batch_size=2 \
#         --output_dir checkpoint_zju_debug_dabrel \


#wandb24
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# torchrun --nproc_per_node=8 \
#     --standalone \
#     --nnodes=1 \
#     cmh_dab_rel_main.py \
#         --dataset vg \
#         --img_folder data/vg/images/ \
#         --ann_path data/vg/ \
#         --modelname dab_detr \
#         --batch_size=2 \
#         --resume ./checkpoint_zju_debug_dabrel/checkpoint0034.pth \
#         --output_dir checkpoint_zju_debug_dabrel \

# export CUDA_VISIBLE_DEVICES=1,2,
# torchrun --nproc_per_node=2 \
#     --standalone \
#     --nnodes=1 \
#     cmh_dab_rel_main.py \
#         --dataset vg \
#         --img_folder data/vg/images/ \
#         --ann_path data/vg/ \
#         --modelname dab_detr \
#         --batch_size=2 \
#         --resume ./checkpoint_zju_debug_dabrel/checkpoint0049.pth \
#         --output_dir checkpoint_zju_debug_dabrel \
#         --eval




# export CUDA_VISIBLE_DEVICES=3
# torchrun --nproc_per_node=1 \
#     --standalone \
#     --nnodes=1 \
#     cmh_dab_rel_main.py \
#         --dataset vg \
#         --img_folder data/vg/images/ \
#         --ann_path data/vg/ \
#         --modelname dab_detr \
#         --batch_size=2 \
#         --resume ./checkpoint_zju_debug_dabrel/checkpoint0119.pth \
#         --output_dir checkpoint_zju_debug_dabrel \
#         --eval

#====================================================================



# 将zju-3090的dabrel放过来跑
#====================================================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
torchrun --nproc_per_node=2 \
         --standalone \
         --nnodes=1 \
        cmh_dab_rel_main.py \
            --dataset vg \
            --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
            --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
            --modelname dab_detr \
            --batch_size=8 \
            --output_dir checkpoint_uts_debug_dabrel \
            --epochs 50 \
            --lr_drop 40 \
            --random_refpoints_xy \
#====================================================================
