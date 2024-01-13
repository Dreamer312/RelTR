# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from .util import box_ops
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
            'pad_size': pad_size,
            'bid':known_bid
        }
    else:  # no dn for inference
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

    input_query_label = input_query_label.transpose(0, 1)
    input_query_bbox = input_query_bbox.transpose(0, 1)

    return input_query_label, input_query_bbox, attn_mask, mask_dict




def prepare_for_dn_tri(dn_args, embedweight_triplets, batch_size, training, num_queries, num_classes, hidden_dim, label_enc, mask_dict_entity, input_query_label_entity):
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

    refpoint_emb = embedweight_triplets.repeat(num_patterns, 1)  # [300, 4]
    

    if training:
        target_sub_boxes, target_sub_labels, target_obj_boxes, target_obj_labels, batch_idx, known_num, rel_labels = extract_sub_obj_tensors(targets)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_labels_sub = target_sub_labels.repeat(scalar, 1).view(-1)
        known_labels_obj = target_obj_labels.repeat(scalar, 1).view(-1)
        known_bboxs_sub = target_sub_boxes.repeat(scalar, 1)
        known_bboxs_obj = target_obj_boxes.repeat(scalar, 1)
        known_rel = rel_labels.repeat(scalar, 1).view(-1)

        #add noise
        known_labels_expaned_sub = known_labels_sub.clone() 
        known_bbox_expand_sub = known_bboxs_sub.clone() 
        known_labels_expaned_obj = known_labels_obj.clone() 
        known_bbox_expand_obj = known_bboxs_obj.clone() 

        # noise on the label
        if label_noise_scale > 0:
            p_sub = torch.rand_like(known_labels_expaned_sub.float()) #[55] 生成的随机数遵循均匀分布，范围在 [0, 1)
            p_obj = torch.rand_like(known_labels_expaned_obj.float())

            chosen_indice_sub = torch.nonzero(p_sub < (label_noise_scale)).view(-1)
            chosen_indice_obj = torch.nonzero(p_obj < (label_noise_scale)).view(-1)

            new_label_sub = torch.randint_like(chosen_indice_sub, 0, num_classes) #这些随机整数的取值范围是从 0 到 num_classes（不包含 num_classes 151）
            new_label_obj = torch.randint_like(chosen_indice_obj, 0, num_classes)


            known_labels_expaned_sub.scatter_(0, chosen_indice_sub, new_label_sub)
            known_labels_expaned_obj.scatter_(0, chosen_indice_obj, new_label_obj)
           

        # noise on the box
        if box_noise_scale > 0:
            diff_sub = torch.zeros_like(known_bbox_expand_sub) 
            diff_sub[:, :2] = known_bbox_expand_sub[:, 2:] / 2 # bbox 中心點坐標: w/2,h/2   将bbox宽高的1/2给diff作为中心坐标
            diff_sub[:, 2:] = known_bbox_expand_sub[:, 2:] # bbox 寬高: w,h
            known_bbox_expand_sub += torch.mul((torch.rand_like(known_bbox_expand_sub) * 2 - 1.0), diff_sub).cuda() * box_noise_scale
            known_bbox_expand_sub = known_bbox_expand_sub.clamp(min=0.0, max=1.0)

            diff_obj = torch.zeros_like(known_bbox_expand_obj) 
            diff_obj[:, :2] = known_bbox_expand_obj[:, 2:] / 2 
            diff_obj[:, 2:] = known_bbox_expand_obj[:, 2:] 
            known_bbox_expand_obj += torch.mul((torch.rand_like(known_bbox_expand_obj) * 2 - 1.0), diff_obj).cuda() * box_noise_scale
            known_bbox_expand_obj = known_bbox_expand_obj.clamp(min=0.0, max=1.0)

        noisy_sub_label = known_labels_expaned_sub.long().to('cuda')
        noisy_obj_label = known_labels_expaned_obj.long().to('cuda')

        input_label_embed_sub = label_enc(noisy_sub_label)
        input_label_embed_obj = label_enc(noisy_obj_label)

        indicator1 = torch.ones([input_label_embed_sub.shape[0], 1]).cuda()

        input_label_embed_sub = torch.cat([input_label_embed_sub, indicator1], dim=1)
        input_label_embed_obj = torch.cat([input_label_embed_obj, indicator1], dim=1)

        input_bbox_embed_sub = inverse_sigmoid(known_bbox_expand_sub)
        input_bbox_embed_obj = inverse_sigmoid(known_bbox_expand_obj)

        #padding
        single_pad = int(max(known_num))  #这个batch里面 最多目标数量
        pad_size = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()   
        padding_bbox = torch.zeros(pad_size, 4).cuda() 
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)

 
        input_query_label_sub = input_query_label.clone()
        input_query_label_obj = input_query_label.clone()

        input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
        input_query_bbox_sub = input_query_bbox.clone()
        input_query_bbox_obj = input_query_bbox.clone()

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            # 计算出去噪任务中真实有效的(非 padding 的) queries 对应的索引
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid):
            input_query_label_sub[(known_bid.long(), map_known_indice)] = input_label_embed_sub
            input_query_bbox_sub[(known_bid.long(), map_known_indice)] = input_bbox_embed_sub

            input_query_label_obj[(known_bid.long(), map_known_indice)] = input_label_embed_obj
            input_query_bbox_obj[(known_bid.long(), map_known_indice)] = input_bbox_embed_obj   

        tgt_size = pad_size + num_queries * num_patterns

        attn_mask2 = torch.ones(tgt_size, tgt_size).to('cuda') < 0   # [390,390]
        # match query cannot see the reconstruct
        attn_mask2[pad_size:, :pad_size] = True        #300行90列的长方形设置为True, 目的是让300个query和90个dn query attn_mask为True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                # attn_mask[0:18, 18:90]
                attn_mask2[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                # attn_mask[18*4:18*5, 0:18]
                attn_mask2[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask2[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask2[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        
        
        attn_mask = create_attn_mask(num_queries, num_queries, single_pad=single_pad, num_denoising_groups=scalar)
        expanded_mask1 = attn_mask2.clone()
        expanded_mask2 = attn_mask2.clone()
        expanded_mask3 = attn_mask2.clone()
        expanded_mask4 = attn_mask2.clone()
        # 复制这个结构以构建完整的 [780, 780] 掩码
        # 首先沿着列方向拼接两个掩码
        horizontal_concat1 = torch.cat([expanded_mask1, expanded_mask2], dim=1)
        horizontal_concat2 = torch.cat([expanded_mask3, expanded_mask4], dim=1)

        # 然后沿着行方向拼接两个大掩码
        attn_mask3 = torch.cat([horizontal_concat1, horizontal_concat2], dim=0)
        assert(attn_mask.equal(attn_mask3.cuda()))

        entiy_pad_size = mask_dict_entity['pad_size']
        attn_mask_dea = torch.zeros((tgt_size, input_query_label_entity.size(0)), dtype=torch.bool).cuda()
        attn_mask_dea[:, :entiy_pad_size] = True

        mask_dict = {
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes_sub': (known_labels_sub, known_bboxs_sub),
            'known_lbs_bboxes_obj': (known_labels_obj, known_bboxs_obj),
            'known_rel':known_rel,
            'pad_size': pad_size,
            'bid': known_bid.long()
        }

    else:  # no dn for inference
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        input_query_label_sub = input_query_label.clone()
        input_query_label_obj = input_query_label.clone()
        input_query_bbox_sub = input_query_bbox.clone()
        input_query_bbox_obj = input_query_bbox.clone()
        attn_mask = None
        attn_mask_dea = None
        mask_dict = None

    input_query_label_sub = input_query_label_sub.transpose(0, 1)
    input_query_label_obj = input_query_label_obj.transpose(0, 1)
    input_query_bbox_sub = input_query_bbox_sub.transpose(0, 1)
    input_query_bbox_obj = input_query_bbox_obj.transpose(0, 1)

    return (input_query_label_sub, input_query_label_obj), (input_query_bbox_sub, input_query_bbox_obj), (attn_mask,attn_mask_dea), mask_dict



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


def dn_post_process_tri(outputs_class_sub, outputs_coord_sub, outputs_class_obj, outputs_coord_obj, output_rel, mask_dict):

    
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        # output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]  # [6,bs,90,151]
        # output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]  # [6,bs,90,4]

        output_known_class_sub = outputs_class_sub[:, :, :mask_dict['pad_size'], :]  # [6,bs,pad_size,151]
        output_known_coord_sub = outputs_coord_sub[:, :, :mask_dict['pad_size'], :]  # [6,bs,pad_size,4]
        output_known_class_obj = outputs_class_obj[:, :, :mask_dict['pad_size'], :]  # [6,bs,pad_size,151]
        output_known_coord_obj = outputs_coord_obj[:, :, :mask_dict['pad_size'], :]  # [6,bs,pad_size,4]
        output_known_rel = output_rel[:, :, :mask_dict['pad_size'], :] # [6,bs,pad_size,51]

        outputs_class_sub = outputs_class_sub[:, :, mask_dict['pad_size']:, :]  # [6,bs,300,151]
        outputs_coord_sub = outputs_coord_sub[:, :, mask_dict['pad_size']:, :]  # [6,bs,300,4]
        outputs_class_obj = outputs_class_obj[:, :, mask_dict['pad_size']:, :]  # [6,bs,300,151]
        outputs_coord_obj = outputs_coord_obj[:, :, mask_dict['pad_size']:, :]  # [6,bs,300,4]
        output_rel = output_rel[:, :, mask_dict['pad_size']:, :] # [6,bs,300,51]

        mask_dict['output_known_lbs_bboxes_sub']=(output_known_class_sub,output_known_coord_sub)
        mask_dict['output_known_lbs_bboxes_obj']=(output_known_class_obj,output_known_coord_obj)
        mask_dict['output_known_rel']=output_known_rel
    return outputs_class_sub, outputs_class_obj, outputs_coord_sub, outputs_coord_obj, output_rel

# def prepare_for_loss(mask_dict):
#     """
#     prepare dn components to calculate loss
#     Args:
#         mask_dict: a dict that contains dn information
#     """
#     # [6,bs,90,151] [6,bs,90,4]
#     output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
#     known_labels, known_bboxs = mask_dict['known_lbs_bboxes'] #[260]   [260,4]
#     map_known_indice = mask_dict['map_known_indice'] #[260] 

#     known_indice = mask_dict['known_indice']  #[260] 
    
#     batch_idx = mask_dict['batch_idx']  # [52]
#     bid = batch_idx[known_indice]  #[260] 

#     #===========================================
#     known_bid = mask_dict['know_bid']
#     assert(known_bid.equal(bid))
#     #===========================================

#     if len(output_known_class) > 0:
#         output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)  #[6,260,151]
#         output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)  #[6,260,4]
#     num_tgt = known_indice.numel()  # 260
#     return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


def prepare_for_loss(mask_dict, mask_dict_tri):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    """

    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes'] #Size([6, bs, 100, 151])  Size([6, bs, 100, 4])
    known_labels, known_bboxs = mask_dict['known_lbs_bboxes'] #[260]   [260,4]


    map_known_indice = mask_dict['map_known_indice'] 
    known_indice = mask_dict['known_indice']  
    batch_idx = mask_dict['batch_idx']  

    bid = batch_idx[known_indice]  
    #===========================================
    known_bid = mask_dict['bid']
    assert(known_bid.equal(bid))
    #===========================================

    #Size([6, bs, 100, 151]) -> Size([bs, 100, 6, 151])
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)  #[6,260,151]
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)  #[6,260,4]
    #===========================================
    num_tgt = known_indice.numel()  # 260
    assert(num_tgt==output_known_class.size(1))
    #===========================================
    
    output_known_class_sub, output_known_coord_sub = mask_dict_tri['output_known_lbs_bboxes_sub'] #[6,bs,25,151] [6,bs,25,4]
    output_known_class_obj, output_known_coord_obj = mask_dict_tri['output_known_lbs_bboxes_obj']
    known_labels_sub, known_bboxs_sub = mask_dict_tri['known_lbs_bboxes_sub'] #[55] [55,4]
    known_labels_obj, known_bboxs_obj = mask_dict_tri['known_lbs_bboxes_obj']
    map_known_indice_tri = mask_dict_tri['map_known_indice']
    bid_tri = mask_dict_tri['bid']
    known_rel = mask_dict_tri['known_rel']
    output_known_rel = mask_dict_tri['output_known_rel'] #[6,bs,25,51]

    if len(output_known_class_sub) > 0:
        output_known_class_sub = output_known_class_sub.permute(1, 2, 0, 3)[(bid_tri, map_known_indice_tri)].permute(1, 0, 2) #[6,55,151]
        output_known_coord_sub = output_known_coord_sub.permute(1, 2, 0, 3)[(bid_tri, map_known_indice_tri)].permute(1, 0, 2) #[6,55,4]
        output_known_class_obj = output_known_class_obj.permute(1, 2, 0, 3)[(bid_tri, map_known_indice_tri)].permute(1, 0, 2)
        output_known_coord_obj = output_known_coord_obj.permute(1, 2, 0, 3)[(bid_tri, map_known_indice_tri)].permute(1, 0, 2)
        output_known_rel = output_known_rel.permute(1, 2, 0, 3)[(bid_tri, map_known_indice_tri)].permute(1, 0, 2) #[6,55,51]
    num_tgt_tri = output_known_class_sub.size(1)

    known_labels_all = torch.cat([known_labels, known_labels_sub, known_labels_obj], dim=0)
    known_bboxs_all = torch.cat([known_bboxs, known_bboxs_sub, known_bboxs_obj], dim=0)
    output_known_class_all = torch.cat([output_known_class, output_known_class_sub, output_known_class_obj], dim=1)
    output_known_coord_all = torch.cat([output_known_coord, output_known_coord_sub, output_known_coord_obj], dim=1)

    num_tgt_all = 2*num_tgt_tri + num_tgt
    return known_labels_all, known_bboxs_all, output_known_class_all, output_known_coord_all, output_known_rel, known_rel, num_tgt_all


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

def tgt_loss_relation(src_logits_, tgt_labels_,  focal_alpha, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """

    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_rel': torch.as_tensor(0.).to('cuda'),
            'tgt_rel_error': torch.as_tensor(0.).to('cuda'),
        }
    num_tgt = src_logits_.size(0)
    tgt_labels_ = tgt_labels_.cuda()
    src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)  #[1,260,151]   [1,260]

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)  #[1,260,152]

    target_classes_onehot = target_classes_onehot[:, :, :-1]  #[1,260,151]
    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=focal_alpha, gamma=2) * src_logits.shape[1]

    losses = {'tgt_loss_rel': loss_ce}

    losses['tgt_rel_error'] = 100 - accuracy(src_logits_, tgt_labels_)[0]
    return losses


def compute_dn_loss(mask_dict, mask_dict_tri, training, aux_num, focal_alpha):
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
        output_known_rel, known_rel, num_tgt = prepare_for_loss(mask_dict, mask_dict_tri)

        losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, focal_alpha))
        losses.update(tgt_loss_boxes(output_known_coord[-1], known_bboxs, num_tgt))
        losses.update(tgt_loss_relation(output_known_rel[-1], known_rel, focal_alpha))

    else:
        losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_class_rel'] = torch.as_tensor(0.).to('cuda')

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

                l_dict = tgt_loss_relation(output_known_rel[i], known_rel, focal_alpha)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_rel'] = torch.as_tensor(0.).to('cuda')
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses



# def compute_dn_loss(mask_dict, training, aux_num, focal_alpha):
#     """
#     compute dn loss in criterion
#     Args:
#         mask_dict: a dict for dn information
#         training: training or inference flag
#         aux_num: aux loss number
#         focal_alpha:  for focal loss
#     """
#     losses = {}
#     if training and 'output_known_lbs_bboxes' in mask_dict:
#         known_labels, known_bboxs, output_known_class, output_known_coord, \
#         num_tgt = prepare_for_loss(mask_dict)
#         losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, focal_alpha))
#         losses.update(tgt_loss_boxes(output_known_coord[-1], known_bboxs, num_tgt))
#     else:
#         losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
#         losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
#         losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
#         losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')

#     if aux_num:
#         for i in range(aux_num):
#             # dn aux loss
#             if training and 'output_known_lbs_bboxes' in mask_dict:
#                 l_dict = tgt_loss_labels(output_known_class[i], known_labels, num_tgt, focal_alpha)
#                 l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                 losses.update(l_dict)
#                 l_dict = tgt_loss_boxes(output_known_coord[i], known_bboxs, num_tgt)
#                 l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                 losses.update(l_dict)
#             else:
#                 l_dict = dict()
#                 l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
#                 l_dict['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
#                 l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
#                 l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
#                 l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                 losses.update(l_dict)
#     return losses



def extract_sub_obj_tensors(data):
    sub_boxes = []
    sub_labels = []
    obj_boxes = []
    obj_labels = []
    rel_labels = []
    batch_idx = []
    know_num = []

    for i, item in enumerate(data):
        boxes = item['boxes']
        labels = item['labels']
        rel_annotations = item['rel_annotations']

        num_sub = 0
        for rel in rel_annotations:
            # 提取 subject 和 object 的索引
            sub_idx, obj_idx, rel_label = rel[0].item(), rel[1].item(), rel[2].item()
            # 提取对应的 boxes 和 labels
            sub_boxes.append(boxes[sub_idx])
            sub_labels.append(labels[sub_idx])
            obj_boxes.append(boxes[obj_idx])
            obj_labels.append(labels[obj_idx])
            rel_labels.append(rel_label)
            batch_idx.append(i)
            num_sub += 1
        know_num.append(num_sub)
        
    # 转换列表为张量
    sub_boxes_tensor = torch.stack(sub_boxes)
    sub_labels_tensor = torch.stack(sub_labels)
    obj_boxes_tensor = torch.stack(obj_boxes)
    obj_labels_tensor = torch.stack(obj_labels)
    batch_idx_tensor = torch.tensor(batch_idx)
    rel_tensor = torch.tensor(rel_labels)

    return sub_boxes_tensor, sub_labels_tensor, obj_boxes_tensor, obj_labels_tensor, batch_idx_tensor, know_num, rel_tensor




def create_attn_mask(num_subject_queries, num_object_queries, num_denoising_groups, single_pad):
    total_queries = num_subject_queries + num_object_queries  # Total number of queries for both subject and object
    pad_size = num_denoising_groups * single_pad # Total denoising queries for each of subject and object
    mask_size = total_queries + pad_size * 2  # Total size of the mask, including both subject and object denoising groups

    # Initialize mask to False (allowing attention interaction)

    attn_mask = torch.zeros((mask_size, mask_size), dtype=torch.bool).cuda()

    # Subject matching query not interacting with own and object's denoising group
    attn_mask[pad_size : (pad_size + num_subject_queries), : pad_size] = True 
    attn_mask[pad_size : (pad_size + num_subject_queries), (num_subject_queries + pad_size) : (num_subject_queries+ pad_size + pad_size)] = True 

    # Object matching query not interacting with subject's and own denoising group
    attn_mask[(num_subject_queries + 2*pad_size) : mask_size, : pad_size] = True
    attn_mask[(num_subject_queries + 2*pad_size) : mask_size, (num_subject_queries + pad_size) : (num_subject_queries+ pad_size + pad_size)] = True

    # Denoising group interactions
    for i in range(num_denoising_groups):
        if i == 0:
            # attn_mask[0:18, 18:90]
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), (pad_size + num_subject_queries + single_pad * (i + 1)) : (num_subject_queries + 2*pad_size)] = True
            attn_mask[(pad_size + num_subject_queries):(pad_size + num_subject_queries + single_pad * (i + 1)), single_pad * (i + 1):pad_size] = True
            attn_mask[(pad_size + num_subject_queries):(pad_size + num_subject_queries + single_pad * (i + 1)), (pad_size + num_subject_queries + single_pad * (i + 1)) : (num_subject_queries + 2*pad_size)] = True
        if i == num_denoising_groups - 1:
            # attn_mask[18*4:18*5, 0:18]
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            #行不变，列变
            attn_mask[single_pad * i:single_pad * (i + 1), (pad_size + num_subject_queries):(pad_size + num_subject_queries+ single_pad * i)] = True
            #行变，列不变
            attn_mask[(pad_size + num_subject_queries + single_pad * i):(pad_size + num_subject_queries + single_pad * i)+single_pad, :single_pad * i] = True
            #行变，列变
            attn_mask[(pad_size + num_subject_queries + single_pad * i):(pad_size + num_subject_queries + single_pad * i)+single_pad, (pad_size + num_subject_queries):(pad_size + num_subject_queries+ single_pad * i)] = True

        else:
            attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            #行不变，列变
            attn_mask[single_pad * i:single_pad * (i + 1), (pad_size + num_subject_queries + single_pad * (i + 1)) : (2*pad_size + num_subject_queries)] = True
            attn_mask[single_pad * i:single_pad * (i + 1), (pad_size + num_subject_queries) : (pad_size + num_subject_queries + single_pad * i)] = True

            #行变，列不变
            attn_mask[pad_size + num_subject_queries+single_pad * i:pad_size + num_subject_queries+single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            attn_mask[pad_size + num_subject_queries+single_pad * i:pad_size + num_subject_queries+single_pad * (i + 1), :single_pad * i] = True

            #行变，列变
            attn_mask[pad_size + num_subject_queries+single_pad * i:pad_size + num_subject_queries+single_pad * (i + 1), (pad_size + num_subject_queries + single_pad * (i + 1)) : (2*pad_size + num_subject_queries)] = True
            attn_mask[pad_size + num_subject_queries+single_pad * i:pad_size + num_subject_queries+single_pad * (i + 1), (pad_size + num_subject_queries) : (pad_size + num_subject_queries + single_pad * i)] = True
            
    return attn_mask


