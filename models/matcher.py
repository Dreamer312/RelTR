# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
"""
Modules to compute the matching cost between the predicted triplet and ground truth triplet.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network"""

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, iou_threshold: float = 0.7):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.iou_threshold = iou_threshold
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_entities, num_entity_classes] with the entity classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_entities, 4] with the predicted box coordinates
                 "sub_logits":  Tensor of dim [batch_size, num_triplets, num_entity_classes] with the subject classification logits
                 "sub_boxes": Tensor of dim [batch_size, num_triplets, 4] with the predicted subject box coordinates
                 "obj_logits":  Tensor of dim [batch_size, num_triplets, num_entity_classes] with the object classification logits
                 "obj_boxes": Tensor of dim [batch_size, num_triplets, 4] with the predicted object box coordinates
                 "rel_logits":  Tensor of dim [batch_size, num_triplets, num_predicate_classes] with the predicate classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "image_id": Image index
                 "orig_size": Tensor of dim [2] with the height and width
                 "size": Tensor of dim [2] with the height and width after transformation
                 "rel_annotations": Tensor of dim [num_gt_triplet, 3] with the subject index/object index/predicate class
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected entity predictions (in order)
                - index_j is the indices of the corresponding selected entity targets (in order)
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected triplet predictions (in order)
                - index_j is the indices of the corresponding selected triplet targets (in order)
            Subject loss weight (Type: bool) to determine if back propagation should be conducted
            Object loss weight (Type: bool) to determine if back propagation should be conducted
        """
        bs, num_queries = outputs["pred_logits"].shape[:2] # bs 100
        num_queries_rel = outputs["rel_logits"].shape[1] # 200
        alpha = 0.25
        gamma = 2.0

        # for labels in targets:
        #     print(labels["rel_annotations"])   #4张图片分别是4 9 8 10个三元组 [4,3]  [9,3]  [8,3]  [10, 3]  共31个三元组

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # bs=4 torch.Size([400, 152])
        out_bbox = outputs["pred_boxes"].flatten(0, 1) # bs=4 每张图片100个query  torch.Size([400, 4])

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]) # torch.Size([47])  4张图共47个物体
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # torch.Size([47, 4]) 4张图共47个物体bbox

        # Compute the entity classification cost. We borrow the cost function from Deformable DETR (https://arxiv.org/abs/2010.04159)
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] #torch.Size([400, 47])

        # Compute the L1 cost between entity boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen entity boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final entity cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou  # 4个图，每个100query torch.Size([400, 47])
        C = C.view(bs, num_queries, -1).cpu() # torch.Size([4, 100, 47])

        sizes = [len(v["boxes"]) for v in targets]  # [12, 10, 14, 11]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        #[(array([ 4, 24, 27, 28, 46, 53, 60, 63, 68, 74, 87, 96]), array([11,  5,  6,  3,  9,  1,  8, 10,  0,  7,  4,  2])), 
        # (array([ 3,  6, 31, 37, 42, 45, 63, 65, 68, 85]), array([3, 4, 0, 1, 2, 8, 5, 7, 9, 6])), 
        # (array([ 2, 11, 15, 20, 23, 27, 32, 34, 58, 59, 66, 90, 91, 95]), array([ 0, 13,  6,  5,  1,  8,  9, 11,  2, 12,  4,  7,  3, 10])), 
        # (array([ 1, 11, 21, 28, 30, 34, 38, 40, 42, 91, 92]), array([ 8,  4,  2,  0,  3,  1,  9, 10,  6,  5,  7]))]

        # Concat the subject/object/predicate labels and subject/object boxes
        sub_tgt_bbox = torch.cat([v['boxes'][v['rel_annotations'][:, 0]] for v in targets]) # torch.Size([31, 4])
        sub_tgt_ids = torch.cat([v['labels'][v['rel_annotations'][:, 0]] for v in targets]) # torch.Size([31]) 
        #sub_tgt_ids tensor([ 77,  95,  95,  95,   8,  40,  40, 121,  82,  82,  84, 121,  43,  20,
        #20,  72,  53,  77,  78,  78, 111,  58,  78,  78,  78,  78,  78,  78,
        #78,  78,  92], device='cuda:0')

        obj_tgt_bbox = torch.cat([v['boxes'][v['rel_annotations'][:, 1]] for v in targets]) #torch.Size([31, 4])
        obj_tgt_ids = torch.cat([v['labels'][v['rel_annotations'][:, 1]] for v in targets]) #取出这4张图片里面的31个宾语的种类
        # tensor([ 95, 127, 144, 145,  89,   8,   8,   8,   8,   8,   8,   8,   8,  72,
        # 111, 126,  57, 111,  72, 111,  20,  44,  40,  57,  58,  61,  92,  92,
        #  92, 111,  44], device='cuda:0')


        rel_tgt_ids = torch.cat([v["rel_annotations"][:, 2] for v in targets]) 
        #tensor([29, 20, 20, 20, 20, 31, 31, 43, 31, 31, 31, 31, 31, 25, 48, 31, 50, 31,
        #25, 48, 31, 29, 20, 20, 20, 20, 21, 31, 44, 48, 29], device='cuda:0')

        sub_prob = outputs["sub_logits"].flatten(0, 1).sigmoid()  # torch.Size([800, 152]) 每张图200 4张图就是800
        sub_bbox = outputs["sub_boxes"].flatten(0, 1) #torch.Size([800, 4])
        obj_prob = outputs["obj_logits"].flatten(0, 1).sigmoid() # torch.Size([800, 152])
        obj_bbox = outputs["obj_boxes"].flatten(0, 1) # torch.Size([800, 4])
        rel_prob = outputs["rel_logits"].flatten(0, 1).sigmoid() #torch.Size([800, 52])

        # Compute the subject matching cost based on class and box.
        neg_cost_class_sub = (1 - alpha) * (sub_prob ** gamma) * (-(1 - sub_prob + 1e-8).log())
        pos_cost_class_sub = alpha * ((1 - sub_prob) ** gamma) * (-(sub_prob + 1e-8).log())
        cost_sub_class = pos_cost_class_sub[:, sub_tgt_ids] - neg_cost_class_sub[:, sub_tgt_ids] #torch.Size([800, 31])
        cost_sub_bbox = torch.cdist(sub_bbox, sub_tgt_bbox, p=1) #torch.Size([800, 31])
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(sub_bbox), box_cxcywh_to_xyxy(sub_tgt_bbox)) #torch.Size([800, 31])

        # Compute the object matching cost based on class and box.
        neg_cost_class_obj = (1 - alpha) * (obj_prob ** gamma) * (-(1 - obj_prob + 1e-8).log())
        pos_cost_class_obj = alpha * ((1 - obj_prob) ** gamma) * (-(obj_prob + 1e-8).log())
        cost_obj_class = pos_cost_class_obj[:, obj_tgt_ids] - neg_cost_class_obj[:, obj_tgt_ids] # torch.Size([800, 31])
        cost_obj_bbox = torch.cdist(obj_bbox, obj_tgt_bbox, p=1) # torch.Size([800, 31])
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(obj_bbox), box_cxcywh_to_xyxy(obj_tgt_bbox)) #torch.Size([800, 31])

        # Compute the object matching cost only based on class.
        neg_cost_class_rel = (1 - alpha) * (rel_prob ** gamma) * (-(1 - rel_prob + 1e-8).log())
        pos_cost_class_rel = alpha * ((1 - rel_prob) ** gamma) * (-(rel_prob + 1e-8).log())
        cost_rel_class = pos_cost_class_rel[:, rel_tgt_ids] - neg_cost_class_rel[:, rel_tgt_ids] #torch.Size([800, 31])

        # Final triplet cost matrix
        C_rel = self.cost_bbox * cost_sub_bbox + self.cost_bbox * cost_obj_bbox  + \
                self.cost_class * cost_sub_class + self.cost_class * cost_obj_class + 0.5 * cost_rel_class + \
                self.cost_giou * cost_sub_giou + self.cost_giou * cost_obj_giou
        C_rel = C_rel.view(bs, num_queries_rel, -1).cpu() #torch.Size([bs, 200, 31])

        sizes1 = [len(v["rel_annotations"]) for v in targets]
        indices1 = [linear_sum_assignment(c[i]) for i, c in enumerate(C_rel.split(sizes1, -1))]
        #[(array([ 28,  63,  70, 122]), array([1, 3, 0, 2])), 
        # (array([ 23,  68,  70, 102, 112, 126, 130, 148, 186]), array([5, 3, 0, 6, 1, 4, 8, 2, 7])), 
        # (array([ 27,  29,  51,  61,  83, 153, 165, 188]), array([3, 0, 6, 7, 5, 4, 1, 2])), 
        # (array([ 14,  65,  82,  97, 115, 148, 157, 165, 169, 198]), array([4, 8, 9, 1, 6, 7, 3, 2, 0, 5]))]
        
        
        
        # assignment strategy to avoid assigning <background-no_relationship-background > to some good predictions
        sub_weight = torch.ones((bs, num_queries_rel)).to(out_prob.device)  #torch.Size([bs, 200])

        #good_sub_detection torch.Size([800, 47])
        good_sub_detection = torch.logical_and((outputs["sub_logits"].flatten(0, 1)[:, :-1].argmax(-1)[:, None] == tgt_ids),
                                               (box_iou(box_cxcywh_to_xyxy(sub_bbox), box_cxcywh_to_xyxy(tgt_bbox))[0] >= self.iou_threshold))
        for i, c in enumerate(good_sub_detection.split(sizes, -1)):
            sub_weight[i, c.sum(-1)[i*num_queries_rel:(i+1)*num_queries_rel].to(torch.bool)] = 0
            sub_weight[i, indices1[i][0]] = 1

        obj_weight = torch.ones((bs, num_queries_rel)).to(out_prob.device)
        good_obj_detection = torch.logical_and((outputs["obj_logits"].flatten(0, 1)[:, :-1].argmax(-1)[:, None] == tgt_ids),
                                               (box_iou(box_cxcywh_to_xyxy(obj_bbox), box_cxcywh_to_xyxy(tgt_bbox))[0] >= self.iou_threshold))
        for i, c in enumerate(good_obj_detection.split(sizes, -1)):
            obj_weight[i, c.sum(-1)[i*num_queries_rel:(i+1)*num_queries_rel].to(torch.bool)] = 0
            obj_weight[i, indices1[i][0]] = 1

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],\
               [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices1],\
               sub_weight, obj_weight


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, iou_threshold=args.set_iou_threshold)
