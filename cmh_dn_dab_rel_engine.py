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
from models.DNDABRelTR.util import misc as utils
import os

#os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

#暂时是rel的uitl还没有用到
from models.DABRelTR.util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from lib.openimages_evaluation import task_evaluation_sg

def train_one_epoch(model: torch.nn.Module, ema_model, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wandb_logger=None, args=None):

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

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

        if need_tgt_for_training:
            dn_args=(targets, args.scalar, args.label_noise_scale, args.box_noise_scale, args.num_patterns)
            if args.contrastive is not False:
                dn_args += (args.contrastive,)

            outputs, mask_dict, mask_dict_tri = model(samples, dn_args=dn_args)
            loss_dict = criterion(outputs, targets, mask_dict, mask_dict_tri)
        else:
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)

    
        #outputs = model(samples)
        #loss_dict = criterion(outputs, targets)
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
        ema_model.update(model)

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

        if  int(os.environ['LOCAL_RANK']) == 0:
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

def evaluate_rel_batch_sig_baseline(outputs, targets, evaluator, evaluator_list):
    num_select = 600
    sub_prob = outputs['sub_logits'].sigmoid() #[bs,600,151]
    sub_prob = sub_prob.view(outputs['sub_logits'].shape[0], -1)
    obj_prob = outputs['obj_logits'].sigmoid() #[bs,600,151]
    obj_prob = obj_prob.view(outputs['obj_logits'].shape[0], -1) #[bs,90600]

    #todo rel_prob 1是由sub_query 1和 obj_query 1 cat然后过mlp得到的，同理rel_prob 200是由sub_query 200和 obj_query 200得到
    #todo 这里取了topk之后，这些值和sub还有obj就无法对应了，需要改进
    rel_prob = outputs['rel_logits'].sigmoid() #[bs, 600, 51]

    #TODO
    for batch, target in enumerate(targets):
        # recovered boxes with original size
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() 

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}


        # sub_topk_values, sub_topk_indexes  [1,600]
        sub_topk_values, sub_topk_indexes = torch.topk(sub_prob[batch].unsqueeze(0), num_select, dim=1)
        sub_topk_boxes = sub_topk_indexes // outputs['sub_logits'].shape[2] #[1,600]
        sub_labels = sub_topk_indexes % outputs['sub_logits'].shape[2]  #[1,600]
        pred_sub_classes = sub_labels.squeeze(0)
        pred_sub_scores = sub_topk_values.squeeze(0)
        sub_boxes = outputs['sub_boxes'][batch].unsqueeze(0)  # [1,600,4]
        sub_boxes = torch.gather(sub_boxes, 1, sub_topk_boxes.unsqueeze(-1).repeat(1,1,4))
        sub_bboxes_scaled = rescale_bboxes(sub_boxes, torch.flip(target['orig_size'],dims=[0]))#.clone().numpy()
        #sub_bboxes_scaled = sub_bboxes_scaled.squeeze(0)  #[600,4]

        obj_topk_values, obj_topk_indexes = torch.topk(obj_prob[batch].unsqueeze(0), num_select, dim=1)
        obj_topk_boxes = obj_topk_indexes // outputs['obj_logits'].shape[2] #[1,600]
        obj_labels = obj_topk_indexes % outputs['obj_logits'].shape[2]  #[1,600]
        pred_obj_classes = obj_labels.squeeze(0)
        pred_obj_scores = obj_topk_values.squeeze(0)
        obj_boxes = outputs['obj_boxes'][batch].unsqueeze(0)
        obj_boxes = torch.gather(obj_boxes, 1, obj_topk_boxes.unsqueeze(-1).repeat(1,1,4))
        obj_bboxes_scaled = rescale_bboxes(obj_boxes, torch.flip(target['orig_size'],dims=[0]))#.clone().numpy()
        #obj_bboxes_scaled = obj_bboxes_scaled.squeeze(0)

        #rel_scores = outputs['rel_logits'][batch][:, 1:-1].softmax(-1)  rel_scores [200,50]
        rel_scores = rel_prob[batch] #[600, 51]
        # top_scores, top_classes = torch.topk(rel_scores, 1, dim=1)
        # print(top_scores.size())
        # print(top_classes.size())

        unique_sub_query = len(set(sub_topk_boxes.squeeze(0).tolist()))
        unique_obj_query = len(set(obj_topk_boxes.squeeze(0).tolist()))
        
        # print(f'sub_topk_boxes {sub_topk_boxes.size()}')
        # print(f'unique_sub_query  {unique_sub_query}') #unique_sub_query  266
        # print(f'unique_obj_query  {unique_obj_query}') #unique_obj_query  145

        mask = [False] * 600

        # 检查每个索引是否同时存在于sub_topk_boxes和obj_topk_boxes中
        for i in range(600):
            if (i in sub_topk_boxes) and (i in obj_topk_boxes):
                mask[i] = True
        #print(f'mask:{[i for i in enumerate(mask)]}')
        valid_pairs = sum(mask)
        true_indices = [index for index, value in enumerate(mask) if value]
        # print(f'Valid pairs: {valid_pairs}')  #115
        # print(true_indices)

        unique_sub_obj_combinations = []
        combine_sub_classes = []
        combine_sub_scores = []
        combine_sub_bboxes_scaled = []
        combine_obj_classes = []
        combine_obj_scores = []
        combine_obj_bboxes_scaled = []
        combine_rel = []
        combine_top_scores = []
        combine_top_classes = []

        for index in true_indices:  
            
            sub_classes_index = sub_labels[sub_topk_boxes == index]
            sub_scores_index = sub_topk_values[sub_topk_boxes == index]
            sub_bboxes_scaled_index = sub_bboxes_scaled[sub_topk_boxes == index]
            # print(sub_bboxes_scaled_index.size())
            # assert(0)
            
            obj_classes_index = obj_labels[obj_topk_boxes == index]
            obj_scores_index = obj_topk_values[obj_topk_boxes == index]
            obj_bboxes_scaled_index = obj_bboxes_scaled[obj_topk_boxes == index]
            

            assert sub_classes_index.size() == sub_scores_index.size()

            # print(f'sub_classes {sub_classes_index}   sub_scores_index  {sub_scores_index}')
            # print(f'obj_classes {obj_classes_index}   obj_scores_index  {obj_scores_index}')

            for i in range(sub_classes_index.size(0)):
                for k in range(obj_classes_index.size(0)):
                    if sub_classes_index[i] != obj_classes_index[k]:
                        #unique_sub_obj_combinations.append((sub_classes_index[i], obj_classes_index[k], index))
                        combine_sub_classes.append(sub_classes_index[i])
                        combine_sub_scores.append(sub_scores_index[i])
                        combine_sub_bboxes_scaled.append(sub_bboxes_scaled_index[i])
                        combine_obj_classes.append(obj_classes_index[k])
                        combine_obj_scores.append(obj_scores_index[k])
                        combine_obj_bboxes_scaled.append(obj_bboxes_scaled_index[k])
                        combine_rel.append(rel_scores[index])

                        top_scores, top_classes = torch.topk(rel_scores[index], 1, dim=0)
                        combine_top_scores.append(top_scores)
                        combine_top_classes.append(top_classes)

        # print(len(combine_sub_classes))
        # print(len(combine_obj_classes))

        combine_sub_classes = torch.stack(combine_sub_classes, dim=0)
        # print(combine_sub_classes.size())
        # print(combine_sub_classes[:10])
        combine_sub_scores = torch.stack(combine_sub_scores, dim=0)
        combine_sub_bboxes_scaled = torch.stack(combine_sub_bboxes_scaled, dim=0)
        combine_obj_classes = torch.stack(combine_obj_classes, dim=0)
        # print(combine_obj_classes[:10])
        combine_obj_scores = torch.stack(combine_obj_scores, dim=0) 
        combine_obj_bboxes_scaled = torch.stack(combine_obj_bboxes_scaled, dim=0) #torch.Size([5142, 4])
        combine_rel = torch.stack(combine_rel, dim=0) #torch.Size([5142, 51])
        combine_top_scores = torch.stack(combine_top_scores, dim=0)
        combine_top_classes = torch.stack(combine_top_classes,dim=0)


        # print(combine_top_scores.size())
        # print(combine_top_classes.size())
        # print(combine_rel.size())
        # assert(0)


        pred_entry = {'sub_boxes': combine_sub_bboxes_scaled.cpu().clone().numpy(),
                      'sub_classes': combine_sub_classes.cpu().clone().numpy(),
                      'sub_scores': combine_sub_scores.cpu().clone().numpy(),
                      'obj_boxes': combine_obj_bboxes_scaled.cpu().clone().numpy(),
                      'obj_classes': combine_obj_classes.cpu().clone().numpy(),
                      'obj_scores': combine_obj_scores.cpu().clone().numpy(),
                      'rel_scores': rel_scores.cpu().clone().numpy(),
                      'pred_rels':combine_top_scores.squeeze(1).cpu().clone().numpy(),
                      'predicate_scores':combine_top_classes.squeeze(1).cpu().clone().numpy()}
        

        evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)





def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):

    #TODO
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        # pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        # pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)


        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :], dim=1)

        # if evaluator['sgdet'].rel_freq is not None :
        #     counterfact_rel_logits = torch.tensor(evaluator['sgdet'].rel_freq).to(outputs['rel_logits'].device)
        #     rel_scores = torch.softmax(outputs['rel_logits'][batch][:, 1:-1]-counterfact_rel_logits, dim=1)
        # else:
        #     rel_scores = outputs['rel_logits'][batch][:, 1:-1].softmax(-1)
        #rel_scores = outputs['rel_logits'][batch][:, 1:-1].softmax(-1) #[300,50]

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




def evaluate_rel_batch_sig(outputs, targets, evaluator, evaluator_list):
    num_select = 600
    sub_prob = outputs['sub_logits'].sigmoid() #[bs,600,151]
    sub_prob = sub_prob.view(outputs['sub_logits'].shape[0], -1)
    obj_prob = outputs['obj_logits'].sigmoid() #[bs,600,151]
    obj_prob = obj_prob.view(outputs['obj_logits'].shape[0], -1) #[bs,90600]

    #todo rel_prob 1是由sub_query 1和 obj_query 1 cat然后过mlp得到的，同理rel_prob 200是由sub_query 200和 obj_query 200得到
    #todo 这里取了topk之后，这些值和sub还有obj就无法对应了，需要改进
    rel_prob = outputs['rel_logits'].sigmoid() #[bs, 600, 51]

    #TODO
    for batch, target in enumerate(targets):
        # recovered boxes with original size
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() 

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}


        # sub_topk_values, sub_topk_indexes  [1,600]
        sub_topk_values, sub_topk_indexes = torch.topk(sub_prob[batch].unsqueeze(0), num_select, dim=1)
        sub_topk_boxes = sub_topk_indexes // outputs['sub_logits'].shape[2] #[1,600]
        sub_labels = sub_topk_indexes % outputs['sub_logits'].shape[2]  #[1,600]
        pred_sub_classes = sub_labels.squeeze(0)
        pred_sub_scores = sub_topk_values.squeeze(0)
        sub_boxes = outputs['sub_boxes'][batch].unsqueeze(0)  # [1,600,4]
        sub_boxes = torch.gather(sub_boxes, 1, sub_topk_boxes.unsqueeze(-1).repeat(1,1,4))
        sub_bboxes_scaled = rescale_bboxes(sub_boxes.cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        sub_bboxes_scaled = sub_bboxes_scaled.squeeze(0)  #[600,4]

        obj_topk_values, obj_topk_indexes = torch.topk(obj_prob[batch].unsqueeze(0), num_select, dim=1)
        obj_topk_boxes = obj_topk_indexes // outputs['obj_logits'].shape[2] #[1,600]
        obj_labels = obj_topk_indexes % outputs['obj_logits'].shape[2]  #[1,600]
        pred_obj_classes = obj_labels.squeeze(0)
        pred_obj_scores = obj_topk_values.squeeze(0)
        obj_boxes = outputs['obj_boxes'][batch].unsqueeze(0)
        obj_boxes = torch.gather(obj_boxes, 1, obj_topk_boxes.unsqueeze(-1).repeat(1,1,4))
        obj_bboxes_scaled = rescale_bboxes(obj_boxes.cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = obj_bboxes_scaled.squeeze(0)

        #rel_scores = outputs['rel_logits'][batch][:, 1:-1].softmax(-1)  rel_scores [200,50]
        rel_scores = rel_prob[batch] #[600, 51]

        sub_boxes_sorted = torch.gather(sub_boxes, 1, sub_topk_boxes.unsqueeze(-1).repeat(1,1,4))

        pred_sub_classes_len = len(set(sub_topk_boxes.squeeze(0).tolist()))
        print(f'pred_sub_classes_len{pred_sub_classes_len}')
        print(sub_topk_boxes)
        print(sub_labels)
        print(sub_bboxes_scaled[0]==sub_bboxes_scaled[12])
        assert(0)



        ###################################################################A-relation-A
        # mask torch.size[200]
        #创建一个掩码，以排除主体和客体类别相同的预测。这是为了消除不太可能形成实际关系的情况。
        mask = (pred_sub_classes - pred_obj_classes != 0).cpu()
        #如果掩码过滤后的预测数量较少（少于198个），则执行以下操作：
        valid_pair_num = mask.sum()
        print(valid_pair_num)
        if valid_pair_num <= 598: 
            #*================================================================================
            #sub_bboxes_scaled[mask]、pred_sub_classes[mask] 等操作使用掩码过滤出有效的预测。
            sub_bboxes_scaled = sub_bboxes_scaled[mask]
            pred_sub_classes = pred_sub_classes[mask]
            pred_sub_scores = pred_sub_scores[mask]
            obj_bboxes_scaled = obj_bboxes_scaled[mask]
            pred_obj_classes = pred_obj_classes[mask]
            pred_obj_scores = pred_obj_scores[mask]
            rel_scores = rel_scores[mask]
            print(valid_pair_num)
            assert(0)
            #*================================================================================



            #*================================================================================          
            # 1. **计算联合得分**：`pred_sub_scores + pred_obj_scores`：
            #    - 这里，`pred_sub_scores` 和 `pred_obj_scores` 是长度为200的numpy数组，包含了主体和客体预测的概率得分。
            #    - 将两个得分数组相加，意味着对于每个预测的关系，联合得分是由主体和客体的得分共同决定的。

            # 2. **排序得分**：`.sort(descending=True)[1]`：
            #    - 将这些联合得分按降序排序。这里使用的是 `torch.sort`，它返回两个数组：排序后的值和相应的索引。
            #    - `[1]` 取出排序后的索引，这些索引对应于原始得分数组中的位置。

            # 3. **选择补充的索引**：`[: mask.shape[0] - mask.sum()]`：
            #    - `mask` 数组用于过滤掉那些主体和客体类别相同的预测。
            #    - `mask.shape[0]` 是掩码的长度（200），而 `mask.sum()` 是掩码中值为True的元素数量，表示被保留的预测数量。
            #    - `mask.shape[0] - mask.sum()` 计算了被掩码过滤掉的元素数量。
            #    - 通过选择排名靠前的这些索引，我们补充那些被掩码过滤掉的预测。

            # 4. **结果**：`padded_indices` 是一个索引数组，指示了应该从排序后的预测中选择哪些元素来补充被掩码过滤掉的预测。

            # 总之，这个过程的目的是在保留高得分预测的同时，为那些因掩码过滤而缺失的预测提供替代项。这样做可以确保在进行场景图评估时，
            # 有足够的预测来考虑，尤其是在排除了某些可能不合逻辑的主体-客体组合后。
            padded_indices = (pred_sub_scores + pred_obj_scores).sort(descending=True)[1][: mask.shape[0] - mask.sum()].cpu()
            padded_sub_bboxes = sub_bboxes_scaled[padded_indices]
            padded_sub_class = pred_sub_classes[padded_indices]
            padded_sub_scores = pred_sub_scores[padded_indices]
            padded_obj_bboxes = obj_bboxes_scaled[padded_indices]
            padded_obj_class = pred_obj_classes[padded_indices]
            padded_obj_scores = pred_obj_scores[padded_indices]
            padded_rel_scores = rel_scores[padded_indices]


            #*================================================================================ 
            # 这段代码的目的是在处理关系得分（`padded_rel_scores`），以增强第二高的得分并消除最高得分。
            # 这样做可能是为了提高模型预测多样性，减少总是选择最高得分预测的倾向。具体步骤如下：

            # 1. **找到最高得分的索引**：`max_value_indices = torch.max(padded_rel_scores, dim=1)[1]`：
            # - `torch.max(padded_rel_scores, dim=1)` 对每行（即每个关系预测）的关系得分进行操作，返回每行的最大值及其索引。
            # - `[1]` 获取这些最大值的索引，即每行得分最高的关系类别的索引。

            # 2. **遍历最高得分的索引**：`for i, idx in enumerate(max_value_indices)`：
            # - 这个循环遍历每个预测（即每行）的最高得分索引。

            # 3. **找到第二高得分的索引**：`second_max_index = (-padded_rel_scores[i]).sort()[1][1]`：
            # - 通过取 `padded_rel_scores[i]` 的负值并排序，可以找到第二高的得分。
            # - `sort()[1]` 返回排序后的索引，`[1]` 获取第二个元素的索引，即第二高得分的索引。

            # 4. **增强第二高得分**：`padded_rel_scores[i, second_max_index] += padded_rel_scores[i, idx]*0.2`：
            # - 这里将第二高得分增强了最高得分的20%。
            # - 这可能是为了在预测关系时增加多样性，避免模型总是倾向于选择得分最高的关系类别。

            # 5. **消除最高得分**：`padded_rel_scores[i, idx] = 0`：
            # - 将最高得分的关系类别得分设置为0，这样在之后的处理中就不会选择这个关系类别。

            # 总结：这段代码通过增强第二高的关系得分并消除最高得分，试图促使模型在预测关系时不仅仅依赖于最高得分，
            # 而是考虑到得分较高但不是最高的其他关系类别。这可能有助于提高模型预测的多样性和准确性。
            max_value_indices = torch.max(padded_rel_scores, dim=1)[1]
            for i, idx in enumerate(max_value_indices):
                second_max_index = (-padded_rel_scores[i]).sort()[1][1]
                padded_rel_scores[i, second_max_index] += padded_rel_scores[i, idx]*0.2
                padded_rel_scores[i, idx] = 0
            #*================================================================================ 


            
            sub_bboxes_scaled = np.concatenate([sub_bboxes_scaled, padded_sub_bboxes], axis=0)
            pred_sub_classes = torch.cat([pred_sub_classes, padded_sub_class], dim=0)
            pred_sub_scores = torch.cat([pred_sub_scores, padded_sub_scores],dim=0)
            obj_bboxes_scaled = np.concatenate([obj_bboxes_scaled, padded_obj_bboxes], axis=0)
            pred_obj_classes = torch.cat([pred_obj_classes, padded_obj_class], dim=0)
            pred_obj_scores = torch.cat([pred_obj_scores, padded_obj_scores],dim=0)
            rel_scores = torch.cat([rel_scores, padded_rel_scores],dim=0)
            #*================================================================================
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