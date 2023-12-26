#WDB65   尝试200+400 带drop    25个epoch
#====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_9 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=400 \
#             --num_select=200 \
#             --set_cost_class=2 



torchrun --nproc_per_node=1 \
         --standalone \
         --nnodes=1 \
         cmh_dab_rel_main_sep.py \
            --dataset vg \
            --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
            --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
            --modelname dab_detr \
            --batch_size=8 \
            --output_dir ./checkpoint_dab_rel_9_debug \
            --epochs 25 \
            --lr_drop 20 \
            --random_refpoints_xy \
            --dropout=0.1 \
            --num_entities=200 \
            --num_triplets=400 \
            --num_select=200 \
            --set_cost_class=2 \
            --resume="/home/minghach/Data/CMH/DAB-RelTR/RelTR/checkpoints/checkpoint_dab_rel_9/checkpoint0024.pth"



# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_9 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=400 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --resume="/home/minghach/Data/CMH/DAB-RelTR/RelTR/checkpoints/checkpoint_dab_rel_9/checkpoint0024.pth"
#====================================================================