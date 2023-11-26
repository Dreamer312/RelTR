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
from models.DABRelTR.util import misc as utils  #import DABRelTR.util.misc 
from datasets import build_dataset, get_coco_api_from_dataset
# from models import build_model
from models.DABRelTR.DABRelTR import build_DABRelTR
from cmh_dab_rel_engine import train_one_epoch, evaluate
import wandb

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


    parser.add_argument('--rel_loss_coef', default=1, type=float)  # RelTR
    #=======================DAB========================================================
    parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")  # Default values are same
    parser.add_argument('--cls_loss_coef', default=1, type=float, help="loss coefficient for cls")  # DAB-DETR
    parser.add_argument('--focal_alpha', type=float, default=0.25, help="alpha for focal loss")  
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

    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")


    return parser


def build_model_main(args):
    if args.modelname.lower() == 'dab_detr':
        model, criterion, postprocessors = build_DABRelTR(args)
    # elif args.modelname.lower() == 'dab_deformable_detr':
    #     model, criterion, postprocessors = build_dab_deformable_detr(args)
    else:
        raise NotImplementedError

    return model, criterion, postprocessors


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if int(os.environ['LOCAL_RANK']) == 0:
        wandb.init(project="SGG", entity="dreamer0312")

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    model, criterion, postprocessors = build_model_main(args)
    #model, criterion, postprocessors = build_model(args)
    print(model)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # dataset_size = len(dataset_train)
    # train_size = int(0.3 * dataset_size)

    # # 使用切片操作来分割数据集
    # dataset_train = torch.utils.data.Subset(dataset_train, indices=range(train_size))

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    # data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_val = DataLoader(dataset_val, args.batch_size, shuffle=False, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        # del checkpoint['optimizer']
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        print('It is the {}th checkpoint'.format(checkpoint['epoch']))
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, wandb_logger = wandb)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth'] # anti-crash
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            # if coco_evaluator is not None:
            #     (output_dir / 'eval').mkdir(exist_ok=True)
            #     if "bbox" in coco_evaluator.coco_eval:
            #         filenames = ['latest.pth']
            #         if epoch % 50 == 0:
            #             filenames.append(f'{epoch:03}.pth')
            #         for name in filenames:
            #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                        output_dir / "eval" / name)
        stats = coco_evaluator.coco_eval["bbox"].stats
        coco_result = {
                        f"AP": round(stats[0], 3),    # 平均精度 AP (IoU=0.50:0.95) - 所有区域大小
                        f"AP50": round(stats[1], 3),  # 平均精度 AP (IoU=0.50) - 所有区域大小
                        f"AP75": round(stats[2], 3),  # 平均精度 AP (IoU=0.75) - 所有区域大小
                        f"APs": round(stats[3], 3),   # 平均精度 AP (IoU=0.50:0.95) - 小区域
                        f"APm": round(stats[4], 3),   # 平均精度 AP (IoU=0.50:0.95) - 中等区域
                        f"APl": round(stats[5], 3),   # 平均精度 AP (IoU=0.50:0.95) - 大区域
                        f"AR1": round(stats[6], 3),   # 平均召回率 AR (最大检测数=1)
                        f"AR10": round(stats[7], 3),  # 平均召回率 AR (最大检测数=10)
                        f"AR100": round(stats[8], 3), # 平均召回率 AR (最大检测数=100)
                        f"ARs": round(stats[9], 3),   # 平均召回率 AR (最大检测数=100) - 小区域
                        f"ARm": round(stats[10], 3),  # 平均召回率 AR (最大检测数=100) - 中等区域
                        f"ARl": round(stats[11], 3)   # 平均召回率 AR (最大检测数=100) - 大区域
                    }
        if int(os.environ['LOCAL_RANK']) == 0:
            wandb.log(coco_result)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DABRelTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)