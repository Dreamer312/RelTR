import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1,
                       cost_class_dab = 2, 
                       cost_bbox: float = 1, 
                       cost_giou: float = 1, 
                       cost_rel: float = 2,
                       iou_threshold: float = 0.7, 
                       focal_alpha = 0.25):
        super().__init__()
        self.cost_class = cost_class
        self.cost_class_dab = cost_class_dab
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rel = cost_rel
        self.iou_threshold = iou_threshold
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]  #bs, 300
        num_queries_rel = outputs["rel_logits"].shape[1] # 600

        # Part1:matching for entity
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() #[1200,151]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  #[1200,4]
        tgt_ids = torch.cat([v["labels"] for v in targets]) # torch.Size([37])  4张图共37个物体
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # torch.Size([37,4])

        alpha = self.focal_alpha  # 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class_dab * cost_class + self.cost_giou * cost_giou
        
        C = C.view(bs, num_queries, -1).cpu()  #[bs,300,47]
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]


        # Part2:matching for sub and obj
        # Concat the subject/object/predicate labels and subject/object boxes
        sub_tgt_bbox = torch.cat([v['boxes'][v['rel_annotations'][:, 0]] for v in targets]) 
        sub_tgt_ids = torch.cat([v['labels'][v['rel_annotations'][:, 0]] for v in targets]) 

        obj_tgt_bbox = torch.cat([v['boxes'][v['rel_annotations'][:, 1]] for v in targets]) #torch.Size([31, 4])
        obj_tgt_ids = torch.cat([v['labels'][v['rel_annotations'][:, 1]] for v in targets]) #取出这4张图片里面的31个宾语的种类

        rel_tgt_ids = torch.cat([v["rel_annotations"][:, 2] for v in targets]) 


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
                self.cost_class * cost_sub_class + self.cost_class * cost_obj_class + self.cost_rel * cost_rel_class + \
                self.cost_giou * cost_sub_giou + self.cost_giou * cost_obj_giou
        C_rel = C_rel.view(bs, num_queries_rel, -1).cpu() #torch.Size([bs, 200, 31])
        sizes1 = [len(v["rel_annotations"]) for v in targets]
        indices1 = [linear_sum_assignment(c[i]) for i, c in enumerate(C_rel.split(sizes1, -1))]

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
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_class_dab = args.set_cost_class_dab, 
                            cost_bbox=args.set_cost_bbox, 
                            cost_giou=args.set_cost_giou, 
                            iou_threshold=args.set_iou_threshold,
                            focal_alpha=args.focal_alpha)