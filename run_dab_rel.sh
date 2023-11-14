export CUDA_VISIBLE_DEVICES=0,2,4,5 
torchrun --nproc_per_node=4 \
    --standalone \
    --nnodes=1 \
    cmh_dab_rel_main.py \
        --dataset vg \
        --img_folder data/vg/images/ \
        --ann_path data/vg/ \
        --modelname dab_detr \
        --batch_size=4 \
        --output_dir checkpoint_zju_debug_dabrel \