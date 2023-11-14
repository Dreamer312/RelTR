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
import util.misc as utils
from util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from lib.openimages_evaluation import task_evaluation_sg

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    is_main_process = not is_initialized() or get_rank() == 0

    # 在主进程上添加tqdm进度条
    if is_main_process:
        data_loader = tqdm(data_loader)
    




    # min_label = float('inf')
    # max_label = float('-inf')
    # min_rel_anno = float('inf')
    # max_rel_anno = float('-inf')
    # for samples, targets in data_loader:  # 假设data_loader是你的数据加载器

    #     # 更新labels的最小和最大值
    #     min_label = min(min_label, torch.min(targets[0]['labels']).item())
    #     max_label = max(max_label, torch.max(targets[0]['labels']).item())

    #     # 更新rel_annotations的第三个数的最小和最大值
    #     third_elements = targets[0]['rel_annotations'][:, 2]  # 获取所有rel_annotations的第三个数
    #     min_rel_anno = min(min_rel_anno, torch.min(third_elements).item())
    #     max_rel_anno = max(max_rel_anno, torch.max(third_elements).item())
    # print(f'Minimum label: {min_label}')
    # print(f'Maximum label: {max_label}')
    # print(f'Minimum third element in rel_annotations: {min_rel_anno}')
    # print(f'Maximum third element in rel_annotations: {max_rel_anno}')

    # assert(0)


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):


        samples = samples.to(device) # torch.Size([bs, 3, 800, 1280])  mask. torch.Size([bs, 800, 1280])

        # 0:{'boxes': tensor([[0.4970, 0.3808, 0.9140, 0.5694],
        # [0.5310, 0.1228, 0.1740, 0.0747],
        # [0.3780, 0.4448, 0.0960, 0.0498],
        # [0.2710, 0.0907, 0.1540, 0.1032],
        # [0.5330, 0.4591, 0.8980, 0.7189],
        # [0.4580, 0.0783, 0.1880, 0.1352],
        # [0.2520, 0.4039, 0.1040, 0.2313],
        # [0.7370, 0.7224, 0.1740, 0.1993],
        # [0.7250, 0.7082, 0.2620, 0.2278],
        # [0.4880, 0.3256, 0.1120, 0.0819],
        # [0.4320, 0.4751, 0.5560, 0.1815],
        # [0.1010, 0.4288, 0.0460, 0.1530]]), 'labels': tensor([ 22,  76,  77,  90,  95,  95, 127, 130, 144, 145, 147, 147]), 'image_id': tensor([2389878]), 'area': tensor([342470.6562,   8593.3428,   3204.3765,  10581.4590, 424867.5625,
        #  16803.0957,  16003.1699,  22973.2734,  39621.9961,   6160.8228,
        #  66786.6875,   4771.4800]), 'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'orig_size': tensor([281, 500]), 'size': tensor([ 608, 1081]), 'rel_annotations': tensor([[ 2,  4, 29],
        # [ 4,  6, 20],
        # [ 4,  8, 20],
        # [ 4,  9, 20]])}

        # 3:{boxes:[11,4], labels:[11], image_id:[2392873], area:[11], iscrowd, orig_size:[375,500], size:[704,938], relation:torch.Size([10, 3]) }


        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        #outputs:{
        # pred_logits: torch.Size([bs, 100, 152])
        # pred_boxes: torch.Size([bs, 100, 4])
        # sub_logits: torch.Size([bs, 200, 152])
        # sub_boxes:torch.Size([bs, 200, 4])
        # obj_logits: torch.Size([bs, 200, 152])
        # obj_boxes: torch.Size([bs, 200, 4])
        # rel_logits: torch.Size([bs, 200, 52])
        #字典队列，长度5[{...},{内容和上面一样}]
        # }
        loss_dict = criterion(outputs, targets)
        #{'loss_ce': tensor(2.9685, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'class_error': tensor(89.3617, device='cuda:0'), 
        # 'sub_error': tensor(93.5484, device='cuda:0'), 
        # 'obj_error': tensor(100., device='cuda:0'), 
        # 'loss_bbox': tensor(1.1930, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_giou': tensor(1.3322, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'cardinality_error': tensor(192.2500, device='cuda:0'), 
        # 'loss_rel': tensor(4.1154, device='cuda:0', grad_fn=<NllLoss2DBackward0>), 
        # 'rel_error': tensor(96.7742, device='cuda:0'), 
        # 'loss_ce_0': tensor(3.1002, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_bbox_0': tensor(1.1808, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_giou_0': tensor(1.3306, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'cardinality_error_0': tensor(192.2500, device='cuda:0'), 
        # 'loss_rel_0': tensor(3.9999, device='cuda:0', grad_fn=<NllLoss2DBackward0>), 
        #  ...
        #  loss_rel_4:tensor(4.1476, device='cuda:0', grad_fn=<NllLoss2DBackward0>)}


        #{'loss_ce': 1, 
        # 'loss_bbox': 5, 
        # 'loss_giou': 2, 
        # 'loss_rel': 1, 
        # 'loss_ce_0': 1, 
        # 'loss_bbox_0': 5, 
        # 'loss_giou_0': 2, 
        # 'loss_rel_0': 1, 
        # 'loss_ce_1': 1, 
        # 'loss_bbox_1': 5, 
        # 'loss_giou_1': 2, 'loss_rel_1': 1, 'loss_ce_2': 1, 'loss_bbox_2': 5, ...}
        weight_dict = criterion.weight_dict

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

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(rel_error=loss_dict_reduced['rel_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args):
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

    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        samples = samples.to(device)
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

        if args.dataset == 'vg':
            evaluate_rel_batch(outputs, targets, evaluator, evaluator_list)
        else:
            evaluate_rel_batch_oi(outputs, targets, all_results)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    if args.dataset == 'vg':
        evaluator['sgdet'].print_stats()
    else:
        task_evaluation_sg.eval_rel_results(all_results, 100, do_val=True, do_vis=False)

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

# def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):
#     for batch, target in enumerate(targets):
#         target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

#         gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
#                     'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
#                     'gt_boxes': target_bboxes_scaled}

#         sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
#         obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

#         pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
#         pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)
#         rel_scores = outputs['rel_logits'][batch][:,1:-1].softmax(-1)

#         pred_entry = {'sub_boxes': sub_bboxes_scaled,
#                       'sub_classes': pred_sub_classes.cpu().clone().numpy(),
#                       'sub_scores': pred_sub_scores.cpu().clone().numpy(),
#                       'obj_boxes': obj_bboxes_scaled,
#                       'obj_classes': pred_obj_classes.cpu().clone().numpy(),
#                       'obj_scores': pred_obj_scores.cpu().clone().numpy(),
#                       'rel_scores': rel_scores.cpu().clone().numpy()}

#         evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)

#         if evaluator_list is not None:
#             for pred_id, _, evaluator_rel in evaluator_list:
#                 gt_entry_rel = gt_entry.copy()
#                 mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
#                 gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
#                 if gt_entry_rel['gt_relations'].shape[0] == 0:
#                     continue
#                 evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):

    #TODO
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

        # if evaluator['sgdet'].rel_freq is not None :
        #     counterfact_rel_logits = torch.tensor(evaluator['sgdet'].rel_freq).to(outputs['rel_logits'].device)
        #     rel_scores = torch.softmax(outputs['rel_logits'][batch][:, 1:-1]-counterfact_rel_logits, dim=1)
        # else:
        #     rel_scores = outputs['rel_logits'][batch][:, 1:-1].softmax(-1)
        rel_scores = outputs['rel_logits'][batch][:, 1:-1].softmax(-1)
        ###################################################################A-relation-A
        #mask = torch.logical_and((pred_sub_classes - pred_obj_classes != 0).cpu(), torch.logical_and(pred_obj_scores >= 0.002, pred_sub_scores >= 0.002).cpu())
        mask = (pred_sub_classes - pred_obj_classes != 0).cpu()
        if mask.sum() <= 198:
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

            sub_bboxes_scaled = np.concatenate([sub_bboxes_scaled, padded_sub_bboxes], axis=0)
            pred_sub_classes = torch.cat([pred_sub_classes, padded_sub_class], dim=0)
            pred_sub_scores = torch.cat([pred_sub_scores, padded_sub_scores],dim=0)
            obj_bboxes_scaled = np.concatenate([obj_bboxes_scaled, padded_obj_bboxes], axis=0)
            pred_obj_classes = torch.cat([pred_obj_classes, padded_obj_class], dim=0)
            pred_obj_scores = torch.cat([pred_obj_scores, padded_obj_scores],dim=0)
            rel_scores = torch.cat([rel_scores, padded_rel_scores],dim=0)
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



def evaluate_rel_batch_oi(outputs, targets, all_results):

    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

        rel_scores = outputs['rel_logits'][batch][:, :-1].softmax(-1)

        relation_idx = target['rel_annotations'].cpu().numpy()
        gt_sub_boxes = target_bboxes_scaled[relation_idx[:, 0]]
        gt_sub_labels = target['labels'][relation_idx[:, 0]].cpu().clone().numpy()
        gt_obj_boxes = target_bboxes_scaled[relation_idx[:, 1]]
        gt_obj_labels = target['labels'][relation_idx[:, 1]].cpu().clone().numpy()

        img_result_dict = {'sbj_boxes': sub_bboxes_scaled,
                           'sbj_labels': pred_sub_classes.cpu().clone().numpy(),
                           'sbj_scores': pred_sub_scores.cpu().clone().numpy(),
                           'obj_boxes': obj_bboxes_scaled,
                           'obj_labels': pred_obj_classes.cpu().clone().numpy(),
                           'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                           'prd_scores': rel_scores.cpu().clone().numpy(),
                           'image': str(target['image_id'].item())+'.jpg',
                           'gt_sbj_boxes': gt_sub_boxes,
                           'gt_sbj_labels': gt_sub_labels,
                           'gt_obj_boxes': gt_obj_boxes,
                           'gt_obj_labels': gt_obj_labels,
                           'gt_prd_labels': relation_idx[:, 2]
                           }
        all_results.append(img_result_dict)
