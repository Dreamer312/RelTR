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



# 将zju-3090的dabrel放过来跑      相当于300+300组合
#====================================================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#         cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir checkpoint_uts_debug_dabrel \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#====================================================================


# wandb39
#====================================================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir checkpoint_uts_debug_dabrel_2 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --num_triplets=200 \
#             --num_entities=100 \
#             --set_cost_class=2

#wdb44
# export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir checkpoint_uts_debug_dabrel \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --num_triplets=200 \
#             --num_entities=100 \
#             --set_cost_class=2 \
#             --eval \
#             --resume ./checkpoint_uts_debug_dabrel_2/checkpoint.pth \

#wdb45之前忘了设置num_select
# export CUDA_VISIBLE_DEVICES=1
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir checkpoint_uts_debug_dabrel \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --num_triplets=200 \
#             --num_entities=100 \
#             --num_select=100 \
#             --set_cost_class=2 \
#             --eval \
#             --resume ./checkpoint_uts_debug_dabrel_2/checkpoint.pth \


#wdb47
# export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir checkpoint_uts_debug_dabrel \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --num_triplets=200 \
#             --num_select=100 \
#             --num_entities=100 \
#             --set_cost_class=2 \
#             --eval \
#             --resume ./checkpoint_uts_debug_dabrel_2/checkpoint0049.pth \
            
#====================================================================


#wdb48      这个命令和log对不上。 不要了
# #====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir checkpoint_uts_debug_dabrel_3 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --num_triplets=300 \
#             --num_entities=200 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --dropout=0.


# #====================================================================




#wdb53
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
#             --output_dir ./checkpoint_dab_rel_5 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 

#wdb55
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_5 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --resume='./checkpoint_uts_debug_dabrel_3/checkpoint0029.pth'
#====================================================================





#wdb56    尝试200+200 不带drop      
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
#             --output_dir ./checkpoint_dab_rel_6 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=200 \
#             --num_triplets=200 \
#             --num_select=200 \
#             --set_cost_class=2 

# export CUDA_VISIBLE_DEVICES=1
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir ./checkpoint_dab_rel_6 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=200 \
#             --num_triplets=200 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --eval \
#             --resume='/home/minghach/Data/CMH/DAB-RelTR/RelTR/checkpoint_dab_rel_6/checkpoint0029.pth'
#====================================================================




#wdb57    尝试200+200 带drop      
# #====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_7 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=200 \
#             --num_select=200 \
#             --set_cost_class=2 



# #wdb59  
# export CUDA_VISIBLE_DEVICES=1
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir ./checkpoint_dab_rel_7 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=200 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --eval \
#             --resume='/home/minghach/Data/CMH/DAB-RelTR/RelTR/checkpoint_dab_rel_7/checkpoint0029.pth'
# #====================================================================







#WDB63   尝试200+400 带drop  
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
#             --output_dir ./checkpoint_dab_rel_8 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=400 \
#             --num_select=200 \
#             --set_cost_class=2 


#WDB64 
# export CUDA_VISIBLE_DEVICES=1
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_8 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=400 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --eval \
#             --resume='/home/minghach/Data/CMH/DAB-RelTR/RelTR/checkpoint_dab_rel_8/checkpoint0029.pth'
#====================================================================




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
#             --eval \
#             --resume="/home/minghach/Data/CMH/DAB-RelTR/RelTR/checkpoints/checkpoint_dab_rel_9/checkpoint0024.pth"
#====================================================================



#WDB68   尝试200+300 带drop    25个epoch
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
#             --output_dir ./checkpoint_dab_rel_10 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 


# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_10 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --eval \
#             --resume="/home/minghach/Data/CMH/DAB-RelTR/RelTR/checkpoints/checkpoint_dab_rel_10/checkpoint0024.pth"
# #====================================================================






#WDB69    WDB65配置加入colorjitter 尝试200+400 带drop    25个epoch
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
#             --output_dir ./checkpoint_dab_rel_11 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=400 \
#             --num_select=200 \
#             --set_cost_class=2 
#====================================================================


#WDB70    WDB65配置加入colorjitter 尝试200+400 带drop    25个epoch set_cost_class=1
# #====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_11 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=400 \
#             --num_select=200 \
#             --set_cost_class=1
# #====================================================================





#WDB72    WDB65配置加入colorjitter 尝试200+400 带drop    25个epoch entity set_cost_class=2    triplet 1
#C = self.cost_bbox * cost_bbox + 2 * cost_class + self.cost_giou * cost_giou
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
#             --output_dir ./checkpoint_dab_rel_13 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=400 \
#             --num_select=200 \
#             --set_cost_class=1
#====================================================================

#WDB73    WDB65配置加入colorjitter 尝试200+400 带drop    25个epoch entity set_cost_class=2    triplet 1
#C = self.cost_bbox * cost_bbox + 2 * cost_class + self.cost_giou * cost_giou
#self.cost_class * cost_sub_class + self.cost_class * cost_obj_class + 1 * cost_rel_class + \
#====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir ./checkpoint_dab_rel_13 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=400 \
#             --num_select=200 \
#             --set_cost_class=1


#WDB78
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir ./checkpoint_dab_rel_13 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=400 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --eval \
#             --resume="/home/minghach/Data/CMH/DAB-RelTR/RelTR/checkpoint_dab_rel_13/checkpoint0024.pth"
#====================================================================




#WDB77    WDB65配置加入colorjitter 尝试200+400 带drop    25个epoch entity set_cost_class=2    triplet 1
#C = self.cost_bbox * cost_bbox + 2 * cost_class + self.cost_giou * cost_giou
#self.cost_class * cost_sub_class + self.cost_class * cost_obj_class + 1 * cost_rel_class + \
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
#             --output_dir ./checkpoint_dab_rel_wdb76 \
#             --epochs 25 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=200 \
#             --num_select=200 \
#             --set_cost_class=1

#WDB80 继续训练试试
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/CMH/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/CMH/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb76_resume \
#             --epochs 50 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=200 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --resume="/home/minghach/Data/CMH/DAB-RelTR/RelTR/checkpoint_dab_rel_wdb76/checkpoint0024.pth"
#====================================================================

#WDB87
#====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb87 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=100 \
#             --num_triplets=200 \
#             --num_select=100 \
#             --set_cost_class=1 \
#====================================================================










#wdb93 dab-reltr-0.1 修复bug，增加作者matcher的weight
#====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb93 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --set_cost_class_dab=2 \
#             --loss_weight \
#====================================================================



#wdb94 dab-reltr-0.1 修复bug，不用作者的weight，增加T.ColorJitter(.4, .4, .4),
#====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb94 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --set_cost_class_dab=2 \

#wdb95 去除重复eval
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb94 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --set_cost_class_dab=2 \
#             --resume="/home/minghach/Data/cmh/DAB-RelTR/RelTR/checkpoint_dab_rel_wdb94/checkpoint0049.pth" \
#             --eval

#wdb97 用了evaluate_rel_batch_sig_test
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb94 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --set_cost_class_dab=2 \
#             --resume="/home/minghach/Data/cmh/DAB-RelTR/RelTR/checkpoint_dab_rel_wdb94/checkpoint0049.pth" \
#             --eval
#====================================================================


# #WDB100 dab-reltr-0.1 修复bug，不用作者的weight，增加T.ColorJitter(.4, .4, .4),
# #====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb100 \
#             --epochs 70 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --set_cost_class_dab=2 \
#             --resume="/home/minghach/Data/cmh/DAB-RelTR/RelTR/checkpoint_dab_rel_wdb94/checkpoint0049.pth" \
# #====================================================================


#WDB101 dab-reltr-0.1 修复bug，不用作者的weight，增加T.ColorJitter(.4, .4, .4),
# #====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb101 \
#             --epochs 60 \
#             --lr_drop 50 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=200 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --set_cost_class_dab=2 \
# #====================================================================





#wdb102 dab-reltr-0.1 修复bug，不用作者的weight，去除T.ColorJitter(.4, .4, .4),与WDB94对照colorjitter
#====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb102 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --set_cost_class_dab=2 \


# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb102 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=1 \
#             --set_cost_class_dab=2 \
#             --resume="/home/minghach/Data/cmh/DAB-RelTR/RelTR/checkpoint_dab_rel_wdb102/checkpoint0049.pth" \
#             --eval
#====================================================================







#--set_cost_class=2 \
#====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb104 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --set_cost_class_dab=2 \


# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb104 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --set_cost_class_dab=2 \
#             --resume="/home/minghach/Data/cmh/DAB-RelTR/RelTR/checkpoints/checkpoint_dab_rel_wdb104/checkpoint0049.pth" \
#             --eval
#====================================================================



#rel 0.5 * cost_rel_class    这之前的实验都是1
#====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb106 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --set_cost_class_dab=2 \

# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb104 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --set_cost_class_dab=2 \
#             --resume="/home/minghach/Data/cmh/DAB-RelTR/RelTR/checkpoint_dab_rel_wdb106/checkpoint0049.pth" \
#             --eval
#====================================================================


#rel 2 * cost_rel_class   试试rel 2的结果
#====================================================================
# torchrun --nproc_per_node=2 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/minghach/Data/cmh/RelTR/data/vg/images/ \
#             --ann_path /home/minghach/Data/cmh/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_wdb108 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --set_cost_class_dab=2 \
#====================================================================



export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 \
         --standalone \
         --nnodes=1 \
         cmh_dab_rel_main.py \
            --dataset vg \
            --img_folder "/data1/cmh/Datasets/VisualGenome/vg/images/" \
            --ann_path "/data1/cmh/Datasets/VisualGenome/vg/" \
            --modelname dab_detr \
            --batch_size=4 \
            --output_dir ./wdb192_one2many_1 \
            --epochs 50 \
            --lr_drop 40 \
            --random_refpoints_xy \
            --dropout=0.1 \
            --num_entities=200 \
            --num_triplets=300 \
            --num_select=200 \
            --set_cost_class=2 \
            --set_cost_class_dab=2 \