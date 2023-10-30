import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os, sys
from typing import Optional
import datasets
import util.misc as utils
from .datasets import build_dataset, get_coco_api_from_dataset



def get_args_parser():
    parser = argparse.ArgumentParser('DAB-RelTR', add_help=False)
    
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    #=======================DAB========================================================
    parser.add_argument('--modelname', type=str, required=True, choices=['dab_detr', 'dab_deformable_detr'])
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    #=======================DAB========================================================

    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    
    #=======================DAB========================================================
    parser.add_argument('--pe_temperatureH', default=20, type=int, 
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, 
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str, 
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")
    #=======================DAB========================================================

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,             
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=600, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    #=======================DAB========================================================
    parser.add_argument('--num_select', default=300, type=int, 
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int, 
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', 
                        help="Random init the x,y of anchor boxes and freeze them.")
    
    # # for DAB-Deformable-DETR
    # parser.add_argument('--two_stage', default=False, action='store_true', 
    #                     help="Using two stage variant for DAB-Deofrmable-DETR")
    # parser.add_argument('--num_feature_levels', default=4, type=int, 
    #                     help='number of feature levels')
    # parser.add_argument('--dec_n_points', default=4, type=int, 
    #                     help="number of deformable attention sampling points in decoder layers")
    # parser.add_argument('--enc_n_points', default=4, type=int, 
    #                     help="number of deformable attention sampling points in encoder layers")
    #=======================DAB========================================================


    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float, 
                        help="Class coefficient in the matching cost")  #! RelTR: 1, DAB-DETR: 2
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,  # ! Dab没有这个
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float, 
                        help="loss coefficient for bbox L1 loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float, 
                        help="loss coefficient for bbox GIOU loss")

    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--rel_loss_coef', default=1, type=float)  # RelTR
    #=======================DAB========================================================
    parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")  # Default values are same
    parser.add_argument('--cls_loss_coef', default=1, type=float, help="loss coefficient for cls")  # DAB-DETR
    parser.add_argument('--focal_alpha', type=float, default=0.25, help="alpha for focal loss")  # ! DAB-DETR独有
    #=======================DAB========================================================


    # dataset parameters
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--ann_path', default='./data/vg/', type=str)
    parser.add_argument('--img_folder', default='/home/cong/Dokumente/tmp/data/visualgenome/images/', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=12, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    parser.add_argument('--find_unused_params', action='store_true')


    return parser


def build_model_main(args):
    if args.modelname.lower() == 'dab_detr':
        model, criterion, postprocessors = build_DABDETR(args)
    # elif args.modelname.lower() == 'dab_deformable_detr':
    #     model, criterion, postprocessors = build_dab_deformable_detr(args)
    # else:
        raise NotImplementedError

    return model, criterion, postprocessors


def main(args):
    model, criterion, postprocessors = build_model_main(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DABRelTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)