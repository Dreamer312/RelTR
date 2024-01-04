# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes

def prepare_for_dn(dn_args, embedweight, batch_size, training, num_queries, num_classes, hidden_dim, label_enc):
    """
    prepare for dn components in forward function
    Args:
        dn_args: (targets, args.scalar, args.label_noise_scale,
                                                             args.box_noise_scale, args.num_patterns) from engine input
        embedweight: positional queries as anchor
        training: whether it is training or inference
        num_queries: number of queries
        num_classes: number of classes
        hidden_dim: transformer hidden dimenstion
        label_enc: label encoding embedding

    Returns: input_query_label, input_query_bbox, attn_mask, mask_dict
    """
    if training:
        targets, scalar, label_noise_scale, box_noise_scale, num_patterns = dn_args
    else:
        num_patterns = dn_args

    if num_patterns == 0:
        num_patterns = 1
    indicator0 = torch.zeros([num_queries * num_patterns, 1]).cuda()  # [300,1]
    # self.label_enc是#[152, 255]的Embedding层，这里num_classes是151，说明将Embedding最后一个元素拿出来，
    #复制300次,这里最后一个Embedding代表去做普通query，剩下的0-151Embedding代表label
    tgt = label_enc(torch.tensor(num_classes).cuda()).repeat(num_queries * num_patterns, 1) #[300, 255]
    tgt = torch.cat([tgt, indicator0], dim=1) #[300, 256]
    refpoint_emb = embedweight.repeat(num_patterns, 1)  # [300, 4]
    if training:
        #known是bs长度list，每个元素都是全1的tensor，一张图片有几个目标就有几个1
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]

        # know_idx 是known里面非零元素的index，但是因为known里都是全1 tensor，所以就是 [[0,1,2...12],[0,6],[0,17],[0,13]]
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]  # [13, 7, 18, 14]  每张图片里面的目标数
        # you can uncomment this to use fix number of dn queries
        # if int(max(known_num))>0:
        #     scalar=scalar//int(max(known_num))


        #todo can be modified to selectively denosie some label or boxes; also known label prediction
        unmask_bbox = unmask_label = torch.cat(known) # 全1tensor，长度13+7+18+14==52
        labels = torch.cat([t['labels'] for t in targets]) #size[52]
        boxes = torch.cat([t['boxes'] for t in targets]) #size[52,4]

        # batch_idx  13个0   7个1   18个2  14个3
        #tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
        # 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        # 3, 3, 3, 3], device='cuda:0')
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)]) 

        known_indice = torch.nonzero(unmask_label + unmask_bbox) #unmask_label + unmask_bbox是52个2    size[52, 1]
        known_indice = known_indice.view(-1) #[52]     内容是[0,1,2,3...51]
        # add noise
        known_indice = known_indice.repeat(scalar, 1).view(-1) # 5*52=260
        known_labels = labels.repeat(scalar, 1).view(-1)  #[260]         把这个batch的所有物体label复制5遍
        known_bid = batch_idx.repeat(scalar, 1).view(-1)  #[260]
        known_bboxs = boxes.repeat(scalar, 1)  #[260,4]                  把这个batch的所有物体boxes复制5遍
        known_labels_expaned = known_labels.clone() #[260]
        known_bbox_expand = known_bboxs.clone() #[260,4]

        # noise on the label
        if label_noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float()) #[260] 生成的随机数遵循均匀分布，范围在 [0, 1)
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # size[43]
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # size[43] 这些随机整数的取值范围是从 0 到 num_classes（不包含 num_classes 151）
            known_labels_expaned.scatter_(0, chosen_indice, new_label) #[260]    将选中的值随机替换成新的label
        # noise on the box
        if box_noise_scale > 0:
            diff = torch.zeros_like(known_bbox_expand) #[260,4]
            diff[:, :2] = known_bbox_expand[:, 2:] / 2 # bbox 中心點坐標: w/2,h/2   将bbox宽高的1/2给diff作为中心坐标
            diff[:, 2:] = known_bbox_expand[:, 2:] # bbox 寬高: w,h

            #torch.rand_like(known_bbox_expand) * 2 - 1.0: 这一行生成了一个与 known_bbox_expand 形状相同的随机数张量，
            # 随机数在 [-1, 1) 范围内。这是通过首先生成 [0, 1) 范围内的随机数，然后乘以 2 并减去 1 来实现的。
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                           diff).cuda() * box_noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        m = known_labels_expaned.long().to('cuda')  #[260]
        input_label_embed = label_enc(m)  # [260，255] 
        # add dn part indicator
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()  # [260，1] 的全1的列向量
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)  # [260，256] 
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        single_pad = int(max(known_num))  # 18  这个batch里面 最多目标数量
        pad_size = int(single_pad * scalar)  # 5组*18 == 90
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()   #[90,256]
        padding_bbox = torch.zeros(pad_size, 4).cuda() #[90,4]
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)  #[bs,390,256]
        input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)  #[bs,390,4]

        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            # 计算出去噪任务中真实有效的(非 padding 的) queries 对应的索引
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [0,..,12, 0,...,6,0,...,17,0,...,13]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()  #[260]
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries * num_patterns   # [390]
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0   # [390,390]
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True        #300行90列的长方形设置为True, 目的是让300个query和90个dn query attn_mask为True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                # attn_mask[0:18, 18:90]
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                # attn_mask[18*4:18*5, 0:18]
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'know_idx': know_idx,
            'pad_size': pad_size
        }
    else:  # no dn for inference
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

    input_query_label = input_query_label.transpose(0, 1)
    input_query_bbox = input_query_bbox.transpose(0, 1)

    return input_query_label, input_query_bbox, attn_mask, mask_dict


def dn_post_process(outputs_class, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]  # [6,bs,90,151]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]  # [6,bs,90,4]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]  # [6,bs,300,151]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]  # [6,bs,300,4]
        mask_dict['output_known_lbs_bboxes']=(output_known_class,output_known_coord)
    return outputs_class, outputs_coord


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    """
    # [6,bs,90,151] [6,bs,90,4]
    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs = mask_dict['known_lbs_bboxes'] #[260]   [260,4]
    map_known_indice = mask_dict['map_known_indice'] #[260] 

    known_indice = mask_dict['known_indice']  #[260] 

    batch_idx = mask_dict['batch_idx']  # [52]
    bid = batch_idx[known_indice]  #[260] 
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)  #[6,260,151]
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)  #[6,260,4]
    num_tgt = known_indice.numel()  # 260
    return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


def tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt,):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    if len(tgt_boxes) == 0:
        return {
            'tgt_loss_bbox': torch.as_tensor(0.).to('cuda'),
            'tgt_loss_giou': torch.as_tensor(0.).to('cuda'),
        }

    loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

    losses = {}
    losses['tgt_loss_bbox'] = loss_bbox.sum() / num_tgt

    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(tgt_boxes)))
    losses['tgt_loss_giou'] = loss_giou.sum() / num_tgt
    return losses


def tgt_loss_labels(src_logits_, tgt_labels_, num_tgt, focal_alpha, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_ce': torch.as_tensor(0.).to('cuda'),
            'tgt_class_error': torch.as_tensor(0.).to('cuda'),
        }

    src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)  #[1,260,151]   [1,260]

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)  #[1,260,152]

    target_classes_onehot = target_classes_onehot[:, :, :-1]  #[1,260,151]
    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=focal_alpha, gamma=2) * src_logits.shape[1]

    losses = {'tgt_loss_ce': loss_ce}

    losses['tgt_class_error'] = 100 - accuracy(src_logits_, tgt_labels_)[0]
    return losses


def compute_dn_loss(mask_dict, training, aux_num, focal_alpha):
    """
    compute dn loss in criterion
    Args:
        mask_dict: a dict for dn information
        training: training or inference flag
        aux_num: aux loss number
        focal_alpha:  for focal loss
    """
    losses = {}
    if training and 'output_known_lbs_bboxes' in mask_dict:
        known_labels, known_bboxs, output_known_class, output_known_coord, \
        num_tgt = prepare_for_loss(mask_dict)
        losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, focal_alpha))
        losses.update(tgt_loss_boxes(output_known_coord[-1], known_bboxs, num_tgt))
    else:
        losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')

    if aux_num:
        for i in range(aux_num):
            # dn aux loss
            if training and 'output_known_lbs_bboxes' in mask_dict:
                l_dict = tgt_loss_labels(output_known_class[i], known_labels, num_tgt, focal_alpha)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_boxes(output_known_coord[i], known_bboxs, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses