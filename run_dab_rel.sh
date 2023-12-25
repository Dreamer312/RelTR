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



#====================================================================
#wdb37
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# torchrun --nproc_per_node=4 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=4 \
#             --output_dir ./checkpoint_dab_rel \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#====================================================================


# #====================================================================
# #试试150 300的组合
#quiet-dream-51
# #/home/cmh/cmh/projects/detrs/RelTR/wandb/offline-run-20231127_233735-4skmahhx
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# torchrun --nproc_per_node=4 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=4 \
#             --output_dir ./checkpoint_dab_rel_ \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=150 \
#             --num_triplets=300 \
#             --num_select=150 \
#             --set_cost_class=2 

# #====================================================================

#====================================================================
#试试200 300的组合

#./wandb/offline-run-20231128_150558-kq0y5pmg/logs         效果不错！
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# torchrun --nproc_per_node=4 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=4 \
#             --output_dir ./checkpoint_dab_rel_ \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 

# export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir ./checkpoint_dab_rel_ \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --resume="./checkpoint_dab_rel_/checkpoint0019.pth"

# #====================================================================


# #====================================================================
#对照上面，加了drop   效果不好
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# torchrun --nproc_per_node=4 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=4 \
#             --output_dir ./checkpoint_dab_rel_2 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 

#====================================================================






# #====================================================================
#8卡     在21epoch 达到了ap50 23.7
#  wandb sync /home/cmh/cmh/projects/detrs/RelTR/wandb/offline-run-20231130_025739-7af2ixe5
# torchrun --nproc_per_node=8 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=2 \
#             --output_dir ./checkpoint_dab_rel_3 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 

# #====================================================================





# # #====================================================================
# #8卡     测试一下300+300组合
#./wandb/offline-run-20231130_151825-icp1mg36/logs
# torchrun --nproc_per_node=8 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=2 \
#             --output_dir ./checkpoint_dab_rel_3 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=300 \
#             --num_triplets=300 \
#             --num_select=300 \
#             --set_cost_class=2 


# torchrun --nproc_per_node=8 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=2 \
#             --output_dir ./checkpoint_dab_rel_3 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=300 \
#             --num_triplets=300 \
#             --num_select=300 \
#             --set_cost_class=2 \
#             --resume="./checkpoint_dab_rel_3/checkpoint0014.pth"




# torchrun --nproc_per_node=8 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=2 \
#             --output_dir ./checkpoint_dab_rel_3 \
#             --epochs 50 \
#             --lr_drop 40 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=300 \
#             --num_triplets=300 \
#             --num_select=300 \
#             --set_cost_class=2 \
#             --resume="./checkpoint_dab_rel_3/checkpoint0024.pth"
# # #====================================================================





#====================================================================
#======================sgdet============================
#Wdb66
# Epoch21后的结果
# R@20: 0.164913
# R@50: 0.210882
# R@100: 0.237388
# Averaged stats: class_error: 32.02  sub_error: 38.53  obj_error: 10.94  rel_error: 27.51  loss: 15.2183 (15.3308)  loss_bbox: 0.7374 (0.7579)  loss_bbox_0: 0.8407 (0.8481)  loss_bbox_1: 0.7680 (0.7802)  loss_bbox_2: 0.7586 (0.7664)  loss_bbox_3: 0.7405 (0.7601)  loss_bbox_4: 0.7446 (0.7584)  loss_ce: 0.6056 (0.5921)  loss_ce_0: 0.6379 (0.6210)  loss_ce_1: 0.6295 (0.6126)  loss_ce_2: 0.6167 (0.6022)  loss_ce_3: 0.6062 (0.5972)  loss_ce_4: 0.6072 (0.5937)  loss_giou: 0.9353 (0.9786)  loss_giou_0: 1.0767 (1.1213)  loss_giou_1: 0.9698 (1.0076)  loss_giou_2: 0.9473 (0.9886)  loss_giou_3: 0.9357 (0.9807)  loss_giou_4: 0.9389 (0.9792)  loss_rel: 0.1691 (0.1630)  loss_rel_0: 0.1679 (0.1638)  loss_rel_1: 0.1690 (0.1653)  loss_rel_2: 0.1731 (0.1648)  loss_rel_3: 0.1712 (0.1644)  loss_rel_4: 0.1696 (0.1637)  cardinality_error_unscaled: 290.7500 (290.0180)  cardinality_error_0_unscaled: 291.8125 (291.0325)  cardinality_error_1_unscaled: 290.9375 (290.6755)  cardinality_error_2_unscaled: 291.3125 (290.1925)  cardinality_error_3_unscaled: 290.2500 (289.3840)  cardinality_error_4_unscaled: 290.1250 (289.3858)  class_error_unscaled: 28.5858 (29.3251)  loss_bbox_unscaled: 0.1475 (0.1516)  loss_bbox_0_unscaled: 0.1681 (0.1696)  loss_bbox_1_unscaled: 0.1536 (0.1560)  loss_bbox_2_unscaled: 0.1517 (0.1533)  loss_bbox_3_unscaled: 0.1481 (0.1520)  loss_bbox_4_unscaled: 0.1489 (0.1517)  loss_ce_unscaled: 0.6056 (0.5921)  loss_ce_0_unscaled: 0.6379 (0.6210)  loss_ce_1_unscaled: 0.6295 (0.6126)  loss_ce_2_unscaled: 0.6167 (0.6022)  loss_ce_3_unscaled: 0.6062 (0.5972)  loss_ce_4_unscaled: 0.6072 (0.5937)  loss_giou_unscaled: 0.4676 (0.4893)  loss_giou_0_unscaled: 0.5384 (0.5607)  loss_giou_1_unscaled: 0.4849 (0.5038)  loss_giou_2_unscaled: 0.4737 (0.4943)  loss_giou_3_unscaled: 0.4679 (0.4903)  loss_giou_4_unscaled: 0.4695 (0.4896)  loss_rel_unscaled: 0.1691 (0.1630)  loss_rel_0_unscaled: 0.1679 (0.1638)  loss_rel_1_unscaled: 0.1690 (0.1653)  loss_rel_2_unscaled: 0.1731 (0.1648)  loss_rel_3_unscaled: 0.1712 (0.1644)  loss_rel_4_unscaled: 0.1696 (0.1637)  obj_error_unscaled: 28.5448 (29.1000)  rel_error_unscaled: 38.4979 (40.0398)  sub_error_unscaled: 35.6836 (34.4449)
# Accumulating evaluation results...
# DONE (t=11.58s).
# IoU metric: bbox
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.130
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.242
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.124
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.050
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.077
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.176
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.211
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.357
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.139
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.254
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.441


# ./wandb/offline-run-20231201_180218-hwwluhc7/logs   
# torchrun --nproc_per_node=8 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=2 \
#             --output_dir ./checkpoint_dab_rel_4 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \

# export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir ./checkpoint_dab_rel_4 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0. \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --eval \
#             --resume='/home/cmh/cmh/projects/detrs/RelTR/checkpoint_dab_rel_4/checkpoint0029.pth'


            
#====================================================================


#====================================================================
# WDB67
# wandb sync /home/cmh/cmh/projects/detrs/RelTR/wandb/offline-run-20231202_172008-dv0u8l8x
# torchrun --nproc_per_node=8 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=2 \
#             --output_dir ./checkpoint_dab_rel_5 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \


# export CUDA_VISIBLE_DEVICES=0
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=1 \
#             --output_dir ./checkpoint_dab_rel_5 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --eval \
#             --resume='/home/cmh/cmh/projects/detrs/RelTR/checkpoint_dab_rel_5/checkpoint0029.pth'


# export CUDA_VISIBLE_DEVICES=1
# torchrun --nproc_per_node=1 \
#          --standalone \
#          --nnodes=1 \
#          cmh_dab_rel_main.py \
#             --dataset vg \
#             --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
#             --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
#             --modelname dab_detr \
#             --batch_size=8 \
#             --output_dir ./checkpoint_dab_rel_5 \
#             --epochs 30 \
#             --lr_drop 20 \
#             --random_refpoints_xy \
#             --dropout=0.1 \
#             --num_entities=200 \
#             --num_triplets=300 \
#             --num_select=200 \
#             --set_cost_class=2 \
#             --eval \
#             --resume='/home/cmh/cmh/projects/detrs/RelTR/checkpoints/checkpoint_dab_rel_5/checkpoint0024.pth'
#====================================================================





#====================================================================
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 \
         --standalone \
         --nnodes=1 \
         cmh_dab_rel_main_dab_rel_softmax.py \
            --dataset vg \
            --img_folder /home/cmh/cmh/projects/detrs/RelTR/data/vg/images/ \
            --ann_path /home/cmh/cmh/projects/detrs/RelTR/data/vg/ \
            --modelname dab_detr \
            --batch_size=4 \
            --output_dir ./checkpoint_dab_rel_softmax_1 \
            --epochs 30 \
            --lr_drop 20 \
            --random_refpoints_xy \
            --dropout=0.1 \
            --num_entities=200 \
            --num_triplets=300 \
            --num_select=200 \
            --set_cost_class=2 \
#====================================================================