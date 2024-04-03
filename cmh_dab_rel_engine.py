# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import numpy as np
from torch.distributed import is_initialized, get_rank
import torch
from datasets.coco_eval import CocoEvaluator

from tqdm import tqdm
from models.DABRelTR.util import misc as utils
import os

#os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

#暂时是rel的uitl还没有用到
from models.DABRelTR.util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from lib.openimages_evaluation import task_evaluation_sg

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wandb_logger=None):
    model.train()
    # criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000

    # is_main_process = not is_initialized() or get_rank() == 0




    # 在主进程上添加tqdm进度条
    # if is_main_process:
    #     #data_loader = tqdm(data_loader)
    #     wandb.init(project="SGG", entity="dreamer0312")

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):


        samples = samples.to(device) # torch.Size([4, 3, h:800, w:1280]))  mask. torch.Size([bs, 800, 1280])

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #targets是list，长度是bs
        # 0:{'boxes': tensor([[0.1987, 0.7031, 0.3291, 0.2057],
    #     [0.2383, 0.5202, 0.4180, 0.9596],
    #     [0.9409, 0.8132, 0.1182, 0.1029],
    #     [0.8496, 0.8236, 0.1191, 0.0768],
    #     [0.3140, 0.1445, 0.1982, 0.2057],
    #     [0.3022, 0.6758, 0.1182, 0.1484],
    #     [0.3291, 0.6549, 0.1152, 0.1484],
    #     [0.4512, 0.6107, 0.1582, 0.1016],
    #     [0.5903, 0.3848, 0.0205, 0.2279],
    #     [0.1509, 0.1068, 0.1475, 0.0469],
    #     [0.2354, 0.6283, 0.3398, 0.7435],
    #     [0.5806, 0.3783, 0.1475, 0.2565],
    #     [0.2549, 0.4629, 0.0586, 0.1107],
    #     [0.2632, 0.5189, 0.4658, 0.9544]], device='cuda:0'), 'labels': tensor([  3,  20,  49,  49,  57,  58,  59,  97,  99, 105, 111, 115,  77,  78],
    #    device='cuda:0'), 'image_id': tensor([498334], device='cuda:0'), 'area': tensor([ 69330.7344, 410723.9688,  12446.6152,   9372.3965,  41763.0234,
    #      17960.9375,  17515.6250,  16453.1250,   4785.1562,   7078.1250,
    #     258734.3906,  38733.0742,   6640.6250, 455261.7188], device='cuda:0'), 'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0'), 'orig_size': tensor([ 768, 1024], device='cuda:0'), 'size': tensor([ 800, 1280], device='cuda:0'), 'rel_annotations': tensor([[ 1,  0, 20],
    #     [11,  8, 50],
    #     [12, 10, 31],
    #     [13,  5, 20],
    #     [13,  7, 21]], device='cuda:0')}
    #     ............
    #     3:{{'boxes': size[6,4], 'labels': tensor([ 22,  26,  64, 130, 142,  26]), 'image_id': tensor([498337]), 'area': tensor([106005.2734,  23460.5098,  39151.1641,   3074.5740,  19535.8594,
        #   25470.5020]), 'iscrowd': tensor([0, 0, 0, 0, 0, 0], device='cuda:0'), 'orig_size': tensor([276, 467], device='cuda:0'), 'size': tensor([ 800, 1280], device='cuda:0'), 
        # 'rel_annotations': tensor([[ 0,  2, 29],[ 1,  0, 29], [ 2,  0, 29]], device='cuda:0')}}







        outputs = model(samples)
        #outputs:{
        # pred_logits: torch.Size([bs, 300, 151])
        # pred_boxes: torch.Size([bs, 300, 4])
        # sub_logits: torch.Size([bs, 600, 151])
        # sub_boxes:torch.Size([bs, 600, 4])
        # obj_logits: torch.Size([bs, 600, 151])
        # obj_boxes: torch.Size([bs, 600, 4])
        # rel_logits: torch.Size([bs, 600, 51])
        #字典队列，长度5[{...},{内容和上面一样}]
        # }

        loss_dict = criterion(outputs, targets)
        # print(loss_dict.keys())
        weight_dict = criterion.weight_dict
        # print(weight_dict.keys())
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name, 'has no grad')
        # assert(0)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(rel_error=loss_dict_reduced['rel_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        #print(loss_dict_reduced_scaled)
        #print(loss_dict_reduced_unscaled)
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        # 使用os.environ.get()获取'WORLD_SIZE'，如果未设置则默认为1
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if local_rank == 0 and world_size > 1:
            wandb_logger.log({
                "loss": loss_value,
                "class_error": loss_dict_reduced['class_error'],
                "sub_error": loss_dict_reduced['sub_error'],
                "obj_error": loss_dict_reduced['obj_error'],
                "rel_error": loss_dict_reduced['rel_error'],
                "lr": optimizer.param_groups[0]["lr"],
                **loss_dict_reduced_scaled
            })


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args, wandb_logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # initilize evaluator
    # TODO merge evaluation programs
    if args.dataset == 'vg':
        evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
        if args.eval:
            evaluator_list = []
            for index, name in enumerate(data_loader.dataset.rel_categories):
                if index == 0:
                    continue
                evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        else:
            evaluator_list = None
    else:
        all_results = []

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # coco_evaluator_sub = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator_obj = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        samples = samples.to(device)
        #* targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets] DAB版
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(rel_error=loss_dict_reduced['rel_error'])

        if  int(os.environ['LOCAL_RANK']) == 0:
            wandb_logger.log({
                "eval_class_error": loss_dict_reduced['class_error'],
                "eval_sub_error": loss_dict_reduced['sub_error'],
                "eval_obj_error": loss_dict_reduced['obj_error'],
                "eval_rel_error": loss_dict_reduced['rel_error'],
            })


        #SGG eval
        if args.dataset == 'vg':
            evaluate_rel_batch(outputs, targets, evaluator, evaluator_list)
        else:
            evaluate_rel_batch_oi(outputs, targets, all_results)


        # 标准的coco eval  与dab一样
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        #results_sub = postprocess_sub(outputs, orig_target_sizes)

    if args.dataset == 'vg':
        #evaluator['sgdet'].print_stats()
        sgg_stats = evaluator['sgdet'].print_stats()
    else:
        task_evaluation_sg.eval_rel_results(all_results, 100, do_val=True, do_vis=False)
    

    
    if  int(os.environ['LOCAL_RANK']) == 0:
        wandb_logger.log(sgg_stats)


    if args.eval and args.dataset == 'vg':
        calculate_mR_from_evaluator_list(evaluator_list, 'sgdet')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    return stats, coco_evaluator



def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):

    #TODO
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :], dim=1)

        rel_scores = outputs['rel_logits'][batch].sigmoid() #[300, 51]

        ###################################################################A-relation-A
        #mask = torch.logical_and((pred_sub_classes - pred_obj_classes != 0).cpu(), torch.logical_and(pred_obj_scores >= 0.002, pred_sub_scores >= 0.002).cpu())
        mask = (pred_sub_classes - pred_obj_classes != 0).cpu()
        if mask.sum() <= (rel_scores.size(0)-2):
            # assert(0)
            sub_bboxes_scaled = sub_bboxes_scaled[mask]
            pred_sub_classes = pred_sub_classes[mask]
            pred_sub_scores = pred_sub_scores[mask]
            obj_bboxes_scaled = obj_bboxes_scaled[mask]
            pred_obj_classes = pred_obj_classes[mask]
            pred_obj_scores = pred_obj_scores[mask]
            rel_scores = rel_scores[mask]

            padded_indices = (pred_sub_scores + pred_obj_scores).sort(descending=True)[1][: mask.shape[0] - mask.sum()].cpu()
            padded_sub_bboxes = sub_bboxes_scaled[padded_indices]
            padded_sub_class = pred_sub_classes[padded_indices]
            padded_sub_scores = pred_sub_scores[padded_indices]
            padded_obj_bboxes = obj_bboxes_scaled[padded_indices]
            padded_obj_class = pred_obj_classes[padded_indices]
            padded_obj_scores = pred_obj_scores[padded_indices]
            padded_rel_scores = rel_scores[padded_indices]
            max_value_indices = torch.max(padded_rel_scores, dim=1)[1]
            for i, idx in enumerate(max_value_indices):
                second_max_index = (-padded_rel_scores[i]).sort()[1][1]
                padded_rel_scores[i, second_max_index] += padded_rel_scores[i, idx]*0.2
                padded_rel_scores[i, idx] = 0

            if mask.sum() < (rel_scores.size(0)-2):
                sub_bboxes_scaled = np.concatenate([sub_bboxes_scaled, padded_sub_bboxes], axis=0)
                pred_sub_classes = torch.cat([pred_sub_classes, padded_sub_class], dim=0)
                pred_sub_scores = torch.cat([pred_sub_scores, padded_sub_scores],dim=0)
                obj_bboxes_scaled = np.concatenate([obj_bboxes_scaled, padded_obj_bboxes], axis=0)
                pred_obj_classes = torch.cat([pred_obj_classes, padded_obj_class], dim=0)
                pred_obj_scores = torch.cat([pred_obj_scores, padded_obj_scores],dim=0)
                rel_scores = torch.cat([rel_scores, padded_rel_scores],dim=0)

            # sub_bboxes_scaled = np.concatenate([sub_bboxes_scaled, padded_sub_bboxes], axis=0)
            # pred_sub_classes = torch.cat([pred_sub_classes, padded_sub_class], dim=0)
            # pred_sub_scores = torch.cat([pred_sub_scores, padded_sub_scores],dim=0)
            # obj_bboxes_scaled = np.concatenate([obj_bboxes_scaled, padded_obj_bboxes], axis=0)
            # pred_obj_classes = torch.cat([pred_obj_classes, padded_obj_class], dim=0)
            # pred_obj_scores = torch.cat([pred_obj_scores, padded_obj_scores],dim=0)
            # rel_scores = torch.cat([rel_scores, padded_rel_scores],dim=0)
        ###################################################################A-relation-A

        #
        pred_entry = {'sub_boxes': sub_bboxes_scaled,
                      'sub_classes': pred_sub_classes.cpu().clone().numpy(),
                      'sub_scores': pred_sub_scores.cpu().clone().numpy(),
                      'obj_boxes': obj_bboxes_scaled,
                      'obj_classes': pred_obj_classes.cpu().clone().numpy(),
                      'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                      'rel_scores': rel_scores.cpu().clone().numpy()}

        evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)

