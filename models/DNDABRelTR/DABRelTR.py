import math
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn

#DAB misc
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .backbone import build_backbone
from .dabrel_transformer import build_transformer
from .matcher import build_matcher
from .util import box_ops

from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, weights=None):
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
    prob = inputs.sigmoid()  #torch.Size([bs, 300, 91])
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")        
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # loss torch.Size([bs, 800, 151])
    # weights torch.Size([bs, 800])
    

    if weights is not None:
        loss = loss * weights.unsqueeze(-1)


    return loss.mean(1).sum() / num_boxes





class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    


class DABRelTR(nn.Module):
    """ This is the DAB-DETR module that performs object detection """
    def __init__(self,  backbone, 
                        transformer, 
                        num_classes,
                        num_rel_classes,
                        num_queries,
                        num_triplets, 
                        num_dec_layers,
                        aux_loss=False, 
                        iter_update=True,
                        query_dim=4, 
                        bbox_embed_diff_each_layer=False,
                        random_refpoints_xy=False,
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for False. (shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
            

        """
        #=============================一致的地方================================================
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        #hidden_dim = 256 if transformer==None else transformer.d_model
        self.hidden_dim = hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        #=============================================================================


        #?=============================DN================================================
        # leave one dim for indicator
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  #[152, 255]
        self.num_classes = num_classes
        #?=============================DN================================================


        #=============================DAB================================================

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(num_dec_layers)])
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.iter_update = iter_update
        #默认是True
        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        # setting query dim
        self.query_dim = query_dim
        assert query_dim in [2, 4]

        self.refpoint_embed = nn.Embedding(num_queries, query_dim) #Embedding(300, 4)

                
        #===========================我加的=========================================
        #另外的600个subject和600个object的框子
        self.refpoint_embed_triplets = nn.Embedding(num_triplets, query_dim) #Embedding(600, 4)



        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

            self.refpoint_embed_triplets.weight.data[:, :2].uniform_(0,1)
            #self.refpoint_embed_triplets.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed_triplets.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed_triplets.weight.data[:, :2])
            self.refpoint_embed_triplets.weight.data[:, :2].requires_grad = False


            # a = self.refpoint_embed_triplets.weight.data[:, :2].equal(self.refpoint_embed.weight.data[:, :2])
            # print(a)
            # assert(0)

        self.bbox_embed_sub = MLP(hidden_dim, hidden_dim, 4, 3)
        self.bbox_embed_obj = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(self.bbox_embed_sub.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed_sub.layers[-1].bias.data, 0)
        nn.init.constant_(self.bbox_embed_obj.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed_obj.layers[-1].bias.data, 0)

        self.transformer.decoder.bbox_embed_sub = self.bbox_embed_sub
        self.transformer.decoder.bbox_embed_obj = self.bbox_embed_obj


        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.sub_class_embed = nn.Linear(hidden_dim, num_classes)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes)
        self.rel_class_embed = MLP(hidden_dim*2+128, hidden_dim, num_rel_classes, 2)  # num_rel_classes==51

        self.sub_class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.obj_class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.rel_class_embed.layers[-1].bias.data = torch.ones(num_rel_classes) * bias_value
        #===========================我加的=========================================
    
        # init prior_prob setting for focal loss
        
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # import ipdb; ipdb.set_trace()
        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        #=============================================================================

        #=============================Rel================================================
        # self.entity_embed = nn.Embedding(num_entities, hidden_dim*2)  #torch.Size([100, 512])
        # self.triplet_embed = nn.Embedding(num_triplets, hidden_dim*3) #triplet_embed torch.Size([200, 768])
        self.so_embed = nn.Embedding(2, hidden_dim) # subject and object encoding [2,256]

        # entity prediction
        # 150类的话 索引是0-149   151类的话索引是0-150 实际只用1-150 0不用
        # self.entity_class_embed 进去是256  出来是152
        # self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)  # 传进来的时候就是num_classes 151   label是从1开始的

        # self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # mask head
        self.so_mask_conv = nn.Sequential(torch.nn.Upsample(size=(28, 28)),
                                          nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(32))
        self.so_mask_fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 128))

        # predicate classification
        # 640 256 256 52
        
        #=============================================================================

    #def forward(self, samples: NestedTensor):
    def forward(self, samples: NestedTensor, dn_args=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)  
        src, mask = features[-1].decompose() # src torch.Size([bs, 2048, 25, 40])   torch.Size([bs, 25, 40])

        assert mask is not None
        #=============================DAB================================================
        # default pipeline
        
        embedweight = self.refpoint_embed.weight #[300,4]

        #result:{hs:[6,bs,300,256]
        #        reference:[6,bs,300,4]
        #        hs_sub:[6,bs,600,256]
        #        reference_sub:[6,bs,600,4]
        #        hs_obj:[6,bs,600,256]
        #        reference_obj:[6,bs,600,4]
        #        so_mask:torch.Size([6, bs, 600, 2, 25, 40])    
        # }
        result = self.transformer(self.input_proj(src), 
                                         mask, embedweight, 
                                         self.refpoint_embed_triplets.weight, 
                                         self.so_embed.weight, 
                                         pos[-1])
        # hs torch.Size([6, bs, 300, 256]) 6应该是6层
        # references torch.Size([6, bs, 300, 4])
        
        hs = result["hs"]
        reference = result["reference"]
        if not self.bbox_embed_diff_each_layer:
            reference_before_sigmoid = inverse_sigmoid(reference) # torch.Size([6, bs, 300, 4])
            tmp = self.bbox_embed(hs) #torch.Size([6, bs, 300, 4])
            tmp[..., :self.query_dim] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid() # torch.Size([6, bs, 300, 4])
        else:
            reference_before_sigmoid = inverse_sigmoid(reference)
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = self.bbox_embed[lvl](hs[lvl])
                tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)



        #===========================我加的=========================================
        hs_sub = result["hs_sub"]
        reference_sub = result["reference_sub"]
        reference_sub_before_sigmoid = inverse_sigmoid(reference_sub)
        tmp_sub = self.bbox_embed_sub(hs_sub)
        tmp_sub[..., :self.query_dim] += reference_sub_before_sigmoid
        outputs_coord_sub = tmp_sub.sigmoid() #[6,bs,600,4]


        hs_obj = result["hs_obj"]
        reference_obj = result["reference_obj"]
        reference_obj_before_sigmoid = inverse_sigmoid(reference_obj)
        tmp_obj = self.bbox_embed_obj(hs_obj)
        tmp_obj[..., :self.query_dim] += reference_obj_before_sigmoid
        outputs_coord_obj = tmp_obj.sigmoid() #[6,bs,600,4]


        so_masks = result["so_masks"]
        so_masks = so_masks.detach()
        so_masks = so_masks.view(-1, 2, src.shape[-2],src.shape[-1])
        so_masks = self.so_mask_conv(so_masks) 
        so_masks = so_masks.view(hs_sub.shape[0], hs_sub.shape[1], hs_sub.shape[2],-1) # [6,bs,600,2048]
        so_masks = self.so_mask_fc(so_masks) # [6,bs,600,128]

        outputs_class_sub = self.sub_class_embed(hs_sub) # [6,bs,600,151]
        outputs_class_obj = self.obj_class_embed(hs_obj) # [6,bs,600,151]
        outputs_class_rel = self.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1)) # torch.Size([6, bs, 600, 51])
        #===========================我加的=========================================

        outputs_class = self.class_embed(hs)  # torch.Size([6, bs, 300, 151])



        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
               'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
               'rel_logits': outputs_class_rel[-1]
               }


    #     if self.aux_loss:      #默认使用
    #         out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
    #     # return out
    # #=============================DAB================================================

    # #=============================REL================================================


    #     outputs_class_rel = self.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1)) # torch.Size([6, bs, 200, 52])

    #     out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
    #            'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
    #            'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
    #            'rel_logits': outputs_class_rel[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                    outputs_class_obj, outputs_coord_obj, outputs_class_rel)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj, outputs_class_rel):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                 'rel_logits': g}
                for a, b, c, d, e, f, g in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                               outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                               outputs_class_rel[:-1])]
    
    #=============================REL================================================




class SetCriterion(nn.Module):
    def __init__(self, num_classes, num_rel_classes, matcher, weight_dict, eos_coef, focal_alpha, losses, loss_weight):
        super().__init__()
        self.num_classes = num_classes #151
        self.matcher = matcher
        self.weight_dict = weight_dict      
        self.losses = losses
        self.loss_weight = loss_weight

        #? DAB
        self.focal_alpha = focal_alpha

        #! rel
        # self.eos_coef = eos_coef 
        # empty_weight = torch.ones(self.num_classes + 1) # torch.Size([152]) 
        # empty_weight[-1] = self.eos_coef
        # self.register_buffer('empty_weight', empty_weight)

        self.num_rel_classes = 51 if num_classes == 151 else 31 # Using entity class numbers to adapt rel class numbers
        # empty_weight_rel = torch.ones(num_rel_classes+1)
        # empty_weight_rel[-1] = self.eos_coef
        # self.register_buffer('empty_weight_rel', empty_weight_rel)
        #! rel

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
            """Entity/subject/object Classification loss with focal loss"""
            assert 'pred_logits' in outputs

            # Entity logits and their indices
            pred_logits = outputs['pred_logits']
            idx = self._get_src_permutation_idx(indices[0]) 
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])

            # Prepare target classes for entities
            target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
            target_classes[idx] = target_classes_o

            # Subject and Object logits
            sub_logits = outputs['sub_logits']
            obj_logits = outputs['obj_logits']

            # Subject and Object indices and classes
            rel_idx = self._get_src_permutation_idx(indices[1])
            target_rels_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
            target_relo_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])
            target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64, device=sub_logits.device)
            target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64, device=obj_logits.device)
            target_sub_classes[rel_idx] = target_rels_classes_o
            target_obj_classes[rel_idx] = target_relo_classes_o

            # Concatenate all targets and logits
            target_classes = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)  # [bs, 1500]
            src_logits = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)  # [bs, 1500, 151]

            # One-hot encode targets for focal loss
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]  # Remove the 'no object' class


            # Compute the focal loss
            #

            all_loss_weight = None
            if self.loss_weight:
                all_loss_weight = torch.cat((torch.ones(pred_logits.shape[:2]).to(pred_logits.device), indices[2]*0.5, indices[3]*0.5), dim=-1)
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2, weights=all_loss_weight) * src_logits.shape[1]

            #TODO Weight the loss if necessary
            #! 先不加原版rel的weight了
            # Adjust these weights based on your requirements
            # loss_weight = torch.cat((torch.ones(pred_logits.shape[:2]), indices[2]*0.5, indices[3]*0.5), dim=-1).to(pred_logits.device)
            # losses = {'loss_ce': (loss_ce * loss_weight).sum() / num_boxes}

            losses = {'loss_ce': loss_ce}
            # Calculate classification errors (optional)
            if log:
                losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
                losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
                losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]

            return losses
    # def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
    #     assert 'pred_logits' in outputs

    #     #* Part1 entity loss
    #     pred_logits = outputs['pred_logits'] #torch.Size([bs, 300, 151])
    #     idx = self._get_src_permutation_idx(indices[0]) 
    #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
    #     target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
    #     target_classes[idx] = target_classes_o

    #     target_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2]+1],
    #                                         dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
    #     target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1) # torch.Size([bs, 300, 152])
    #     #target_classes_onehot = target_classes_onehot[:,:,:-1] # 切掉最后一个1，这样就变成了全0向量代表no obj

    #    #* Part2 sub and obj loss
    #     sub_logits = outputs['sub_logits'] #torch.Size([bs, 600, 151])
    #     obj_logits = outputs['obj_logits'] #torch.Size([bs, 600, 151])

    #     rel_idx = self._get_src_permutation_idx(indices[1])
    #     target_rel_sub_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
    #     target_rel_obj_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])
    #     target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64, device=sub_logits.device)
    #     target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64, device=obj_logits.device)
    #     target_sub_classes[rel_idx] = target_rel_sub_classes_o
    #     target_obj_classes[rel_idx] = target_rel_obj_classes_o

    #     target_sub_classes_onehot = torch.zeros([sub_logits.shape[0], sub_logits.shape[1], sub_logits.shape[2]+1],
    #                                              dtype=sub_logits.dtype, layout=sub_logits.layout, device=sub_logits.device)
    #     target_obj_classes_onehot = torch.zeros([obj_logits.shape[0], obj_logits.shape[1], obj_logits.shape[2]+1],
    #                                              dtype=obj_logits.dtype, layout=obj_logits.layout, device=obj_logits.device)
        
    #     target_sub_classes_onehot.scatter_(2, target_sub_classes.unsqueeze(-1), 1)
    #     target_obj_classes_onehot.scatter_(2, target_obj_classes.unsqueeze(-1), 1)


    #     target_classes_onehot1 = torch.cat((target_classes_onehot, target_sub_classes_onehot, target_obj_classes_onehot), dim=1)
    #     target_classes_onehot1 = target_classes_onehot1[:, :, :-1]

    #     target_classes_all = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)  # [bs, 500]
    #     src_logits_all = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)  # [bs, 500, 152]


    #     target_classes_onehot = torch.zeros_like(src_logits_all).scatter_(2, target_classes_all.unsqueeze(-1), 1)
    #     target_classes_onehot = target_classes_onehot[:, :, :-1] 

    #     flag =  target_classes_onehot1.equal(target_classes_onehot)

    #     loss_ce = sigmoid_focal_loss(src_logits_all, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits_all.shape[1]

    #     # TODO Rel这里有个weight  这个weight是从matcher里面产生的，
    #     #loss_weight = torch.cat((torch.ones(pred_logits.shape[:2]).to(pred_logits.device), indices[2]*0.5, indices[3]*0.5), dim=-1)
    #     #losses = {'loss_ce': (loss_ce * loss_weight).sum()/self.empty_weight[target_classes].sum()}

    #     losses = {'loss_ce': loss_ce}

    #     if log:
    #         losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
    #         losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rel_sub_classes_o)[0]
    #         losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_rel_obj_classes_o)[0]
    #     return losses



    def loss_relations(self, outputs, targets, indices, num_boxes, log=True):
        """Compute the predicate classification loss
        """
        assert 'rel_logits' in outputs

        src_logits = outputs['rel_logits']  #[bs,600,51]
        idx = self._get_src_permutation_idx(indices[1])
        target_classes_o = torch.cat([t["rel_annotations"][J,2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1] 
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        #loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses





    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['rel_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the entity/subject/object bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        pred_boxes = outputs['pred_boxes'][idx]
        target_entry_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 0]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        target_relo_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 1]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        rels_boxes = outputs['sub_boxes'][rel_idx]
        relo_boxes = outputs['obj_boxes'][rel_idx]

        src_boxes = torch.cat((pred_boxes, rels_boxes, relo_boxes), dim=0)
        target_boxes = torch.cat((target_entry_boxes, target_rels_boxes, target_relo_boxes), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)



    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices
        num_boxes = sum(len(t["labels"])+len(t["rel_annotations"]) for t in targets) #78=47+31
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()  #[bs,300,91]

        #topk_values [bs,300]   topk_indexes[bs,300]
        #prob.view(out_logits.shape[0], -1) 的作用是将 prob 的形状从 [batch_size, 300, 91] 转换成 [batch_size, 300 * 91]。
        # 这样做的目的是将每个 query 的所有类别概率展平为一个长向量，以便在所有类别中选择最高的 num_select 个概率值。
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values

        # 例如，假设 topk_indexes 中的一个值是 181，表示在展平后的数组中第 182 个元素的概率最高。由于每个 query 有 91 个类别，
        # 所以可以通过以下计算得到原始的 query 索引和类别索引：Query 索引：181 // 91 = 1（整数除法，结果为 1）
        # 类别索引：181 % 91 = 90（求余数，结果为 90）
        # 因此，这个最高概率值对应于第二个 query（索引为 1，因为索引从 0 开始）的90类别（索引为 0）。
        # 通过这种方式，topk_boxes 就能正确表示每个最高概率值所对应的 query 索引。
        topk_boxes = topk_indexes // out_logits.shape[2]  #[bs,300]

        test_index = topk_boxes[0]

        # for query_index in range(test_index.shape[0]):
        #     query_boxes = test_index[query_index]
        #     box_counts = Counter(query_boxes.tolist())
        #     if any(count > 1 for count in box_counts.values()):
        #         print(f"Duplicate detected in batch  query {query_index}")
        # assert(0)




        labels = topk_indexes % out_logits.shape[2]       #[bs,300]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) #[bs, 300, 4]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates+
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # scores:[bs,300]   labels:[bs,300]   boxes:[bs,300,4]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class PostProcessSO(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, type='object') -> None:
        super().__init__()
        self.num_select = num_select
        self.type = type  # 'subject' or 'object'

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        if self.type == 'subject':
            out_logits, out_bbox = outputs['sub_logits'], outputs['sub_boxes']
        elif self.type == 'object':
            out_logits, out_bbox = outputs['obj_logits'], outputs['obj_boxes']
        # ... rest of the code remains the same ...

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()  #[bs,300,91]

        #topk_values [bs,300]   topk_indexes[bs,300]
        #prob.view(out_logits.shape[0], -1) 的作用是将 prob 的形状从 [batch_size, 300, 91] 转换成 [batch_size, 300 * 91]。
        # 这样做的目的是将每个 query 的所有类别概率展平为一个长向量，以便在所有类别中选择最高的 num_select 个概率值。
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values

        # 例如，假设 topk_indexes 中的一个值是 181，表示在展平后的数组中第 182 个元素的概率最高。由于每个 query 有 91 个类别，
        # 所以可以通过以下计算得到原始的 query 索引和类别索引：Query 索引：181 // 91 = 1（整数除法，结果为 1）
        # 类别索引：181 % 91 = 90（求余数，结果为 90）
        # 因此，这个最高概率值对应于第二个 query（索引为 1，因为索引从 0 开始）的90类别（索引为 0）。
        # 通过这种方式，topk_boxes 就能正确表示每个最高概率值所对应的 query 索引。
        topk_boxes = topk_indexes // out_logits.shape[2]  #[bs,300]


        labels = topk_indexes % out_logits.shape[2]       #[bs,300]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) #[bs, 300, 4]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates+
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # scores:[bs,300]   labels:[bs,300]   boxes:[bs,300,4]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results





def build_DNDABRelTR(args):

    num_classes = 151 if args.dataset != 'oi' else 289 # some entity categories in OIV6 are deactivated.
    num_rel_classes = 51 if args.dataset != 'oi' else 31

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)
    model = DABRelTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_rel_classes = num_rel_classes,
        num_queries=args.num_entities,
        num_triplets=args.num_triplets,
        num_dec_layers=args.dec_layers,
        iter_update=True,
        query_dim=4,
        aux_loss=args.aux_loss,
        random_refpoints_xy=args.random_refpoints_xy)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_rel'] = args.rel_loss_coef


    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)


    losses = ['labels', 'boxes', 'cardinality', "relations"]

    criterion = SetCriterion(num_classes, 
                             num_rel_classes, 
                             matcher=matcher, 
                             weight_dict=weight_dict,
                             eos_coef=args.eos_coef,
                             focal_alpha=args.focal_alpha, 
                             losses=losses,
                             loss_weight=args.loss_weight)
    
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    postprocess_sub = PostProcessSO(num_select=args.num_triplets, type='subject')
    postprocess_obj = PostProcessSO(num_select=args.num_triplets, type='object')

    return model, criterion, postprocessors, postprocess_sub, postprocess_obj


