import math
import copy
import os
from typing import Optional, List
from .util.misc import inverse_sigmoid
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .DAB_MHA import MultiheadAttention

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

def gen_sineembed_for_position(pos_tensor, d_model=256):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device) # 0到127
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
    x_embed = pos_tensor[:, :, 0] * scale # torch.Size([300, bs)
    y_embed = pos_tensor[:, :, 1] * scale # torch.Size([300, bs])
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2) #torch.Size([300, bs, 128])
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2) #torch.Size([300, bs, 128])
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2) #torch.Size([300, bs, 128])

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2) #torch.Size([300, bs, 512])
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class Transformer(nn.Module):

    def __init__(self, 
                 d_model=512, 
                 nhead=8, 
                 num_queries=300, 
                 num_encoder_layers=6,
                 num_decoder_layers=6, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="relu", 
                 normalize_before=False,
                 return_intermediate_dec=False, 
                 query_dim=4,
                 keep_query_pos=False, 
                 query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 ):

        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, 
                                                nhead, 
                                                dim_feedforward,
                                                dropout, 
                                                activation, 
                                                normalize_before, #DAB多的
                                                keep_query_pos=keep_query_pos) #DAB多的     其他都一样
        decoder_norm = nn.LayerNorm(d_model) #DAB多的
        decoder_norm_sub = nn.LayerNorm(d_model)
        decoder_norm_obj = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, #
                                          num_decoder_layers,  #
                                          decoder_norm,
                                          decoder_norm_sub,
                                          decoder_norm_obj,
                                          return_intermediate=return_intermediate_dec,#
                                          d_model=d_model, 
                                          query_dim=query_dim, 
                                          keep_query_pos=keep_query_pos, 
                                          query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    #def forward(self, src, mask, refpoint_embed, refpoint_embed_triplets, so_embed, pos_embed):
    def forward(self, src, mask, refpoint_embed, refpoint_embed_triplets, so_embed, pos_embed, tgt, attn_mask=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape #torch.Size([bs, 256, 25, 40])
        src = src.flatten(2).permute(2, 0, 1) # torch.Size([1000, bs, 256])
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # torch.Size([1000, bs, 256])
        #refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1) #[300, bs, 4]
        mask = mask.flatten(1) #torch.Size([bs, 1000])      

        #===================================我加的==============================================  
        
        so_embed = so_embed #[2,256]
        refpoint_embed_triplets = refpoint_embed_triplets.unsqueeze(1).repeat(1, bs, 1) # [600, bs, 4]
        num_queries_triplets = refpoint_embed_triplets.shape[0]
        tgt_triplets = torch.zeros(num_queries_triplets, bs, self.d_model*2, device=refpoint_embed_triplets.device)#torch.Size([600, bs, 512])
        #===================================我加的==============================================  

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) ## torch.Size([1000, bs, 256])

        
        #?=============================DN================================================
        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        result, sub_maps, obj_maps = self.decoder(
                                                    tgt=tgt,
                                                    tgt_triplets=tgt_triplets,
                                                    memory=memory,
                                                    tgt_mask=attn_mask,  
                                                    memory_mask=None,  
                                                    tgt_key_padding_mask=None,  
                                                    memory_key_padding_mask=mask,
                                                    pos=pos_embed,
                                                    so_embed=so_embed,
                                                    refpoints_unsigmoid=refpoint_embed,
                                                    refpoints_unsigmoid_triplets=refpoint_embed_triplets
                                                )
        #?=============================DN================================================

        so_masks = torch.cat((sub_maps.reshape(sub_maps.shape[0], bs, sub_maps.shape[2], 1, h, w),
                              obj_maps.reshape(obj_maps.shape[0], bs, obj_maps.shape[2], 1, h, w)), dim=3)
        result["so_masks"] = so_masks   
        return result


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output) # torch.Size([1000, bs, 256])
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales)

        if self.norm is not None:
            output = self.norm(output)

        return output
    

    

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, norm_sub=None, norm_obj=None, return_intermediate=False, 
                    d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                    modulate_hw_attn=False,
                    bbox_embed_diff_each_layer=False,
                    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

        #===================================我加的==============================================
        self.ref_point_head_sub = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.ref_point_head_obj = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_scale_sub = MLP(d_model, d_model, d_model, 2)
        self.query_scale_obj = MLP(d_model, d_model, d_model, 2)
        self.ref_anchor_head_sub = MLP(d_model, d_model, 2, 2)
        self.ref_anchor_head_obj = MLP(d_model, d_model, 2, 2)

        self.bbox_embed_sub = None
        self.bbox_embed_obj = None

        self.norm_sub = norm_sub
        self.norm_obj = norm_obj

        if not keep_query_pos:
            # 0层由于内容 query 是全0向量没有信息，是必须会相加的，因此这里排除了0层
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].sub_dc_ca_qpos_proj = None

        if not keep_query_pos:
            # 0由于内容 query 是全0向量没有信息，是必须会相加的，因此这里排除了0层
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].obj_dc_ca_qpos_proj = None
        #===================================我加的==============================================


        self.bbox_embed = None #MLP(hidden_dim, hidden_dim, 4, 3) 在DAB init完成初始化
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        # 2层 MLP，输出维度：2，分别用于 x, y 坐标
        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        # 是否要在每层计算交叉注意力前将位置 query 与内容 query 结合(相加)
        # 相加前，会将位置 query 经过 projection(MLP)，即如下的 ca_qpos_proj
        if not keep_query_pos:
            # 第一层由于内容 query 没有位置信息，是必须会相加的，因此这里排除了第一层
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, 
                tgt, 
                tgt_triplets,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                so_embed = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                refpoints_unsigmoid_triplets: Optional[Tensor] = None
                ):
        
        output = tgt  #torch.Size([300, bs, 256]) 全0

        #===================================我加的==============================================
        output_sub, output_obj = torch.split(tgt_triplets, self.d_model, dim=2) # torch.Size([600, bs, 256]) torch.Size([600, bs, 256])
        reference_points_sub = refpoints_unsigmoid_triplets.sigmoid() #torch.Size([600, bs, 4])
        reference_points_obj = refpoints_unsigmoid_triplets.sigmoid()
        ref_points_sub = [reference_points_sub]
        ref_points_obj = [reference_points_obj]
        intermediate_output_sub = []
        intermediate_output_obj = []
        intermediate_submaps = []
        intermediate_objmaps = []
        #===================================我加的==============================================

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid() # torch.Size([300, bs, 4])
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()        

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]     # [num_queries, batch_size, 4]
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model)  #query_sine_embed   torch.Size([300, bs, 512])
            query_pos = self.ref_point_head(query_sine_embed) #torch.Size([300, bs, 256])

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output) # [300,bs,256]
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            #pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2) 前256维是yx，后256维是wh，在这里只取了yx的embed
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation  #query_sine_embed  torch.Size([300, bs, 256])
            #query_sine_embed2 = query_sine_embed.clone()
            # modulated HW attentions
            if self.modulate_hw_attn:
                # 基于当前层的 output 生成 x, y 坐标的调制参数(向量)，对应于 paper 公式中的 w_{q,ref} & h_{q,ref}
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2 torch.Size([300, bs, 2])

                # ref_w = refHW_cond[..., 0] #torch.Size([300, bs])  
                # ref_h = refHW_cond[..., 1] #torch.Size([300, bs])  

                # anchor_w = obj_center[..., 2] #torch.Size([300, bs])  
                # anchor_h = obj_center[..., 3] #torch.Size([300, bs])  

                # factor_x = (ref_w/anchor_w).unsqueeze(-1)
                # query_sine_embed2[..., self.d_model // 2:] *= factor_x
                # factor_y = (ref_h/anchor_h).unsqueeze(-1)
                # query_sine_embed2[..., :self.d_model // 2] *= factor_y
                
                
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1) # x
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1) # y

                # a = query_sine_embed2.equal(query_sine_embed)

            #===================================我加的==============================================
            obj_center_sub = reference_points_sub[..., :self.query_dim] #torch.Size([600, bs, 4])
            query_sine_embed_sub = gen_sineembed_for_position(obj_center_sub, self.d_model)
            obj_center_obj = reference_points_obj[..., :self.query_dim] #torch.Size([600, bs, 4])
            query_sine_embed_obj = gen_sineembed_for_position(obj_center_obj, self.d_model)

            query_pos_sub = self.ref_point_head_sub(query_sine_embed_sub) #torch.Size([600, bs, 256])
            query_pos_obj = self.ref_point_head_obj(query_sine_embed_obj) #torch.Size([600, bs, 256])

            if layer_id == 0:
                pos_transformation_sub = 1
                pos_transformation_obj = 1
            else:
                pos_transformation_sub = self.query_scale_sub(output_sub) # [600,bs,256]
                pos_transformation_obj = self.query_scale_obj(output_obj) # [600,bs,256]

            query_sine_embed_sub = query_sine_embed_sub[...,:self.d_model] * pos_transformation_sub #torch.Size([600, bs, 256])
            query_sine_embed_obj = query_sine_embed_obj[...,:self.d_model] * pos_transformation_obj #torch.Size([600, bs, 256])

            if self.modulate_hw_attn:
                refHW_cond_sub = self.ref_anchor_head_sub(output_sub).sigmoid()
                query_sine_embed_sub[..., self.d_model // 2:] *= (refHW_cond_sub[..., 0] / obj_center_sub[..., 2]).unsqueeze(-1)
                query_sine_embed_sub[..., :self.d_model // 2] *= (refHW_cond_sub[..., 1] / obj_center_sub[..., 3]).unsqueeze(-1)


                refHW_cond_obj = self.ref_anchor_head_obj(output_obj).sigmoid()
                query_sine_embed_obj[..., self.d_model // 2:] *= (refHW_cond_obj[..., 0] / obj_center_obj[..., 2]).unsqueeze(-1)
                query_sine_embed_obj[..., :self.d_model // 2] *= (refHW_cond_obj[..., 1] / obj_center_obj[..., 3]).unsqueeze(-1)



            #same = query_pos_sub.equal(query_pos_obj)  False
            #===================================我加的==============================================

            # output torch.Size([300, bs, 256])
            # output_sub output_obj [600,bs,256]
            # sub_maps obj_maps [bs, 600, 1000]
            output, output_sub, output_obj, sub_maps, obj_maps = layer( output,
                                                                        output_sub,
                                                                        output_obj, 
                                                                        memory, 
                                                                        tgt_mask=tgt_mask, #None
                                                                        memory_mask=memory_mask, #None
                                                                        tgt_key_padding_mask=tgt_key_padding_mask, # None
                                                                        memory_key_padding_mask=memory_key_padding_mask,
                                                                        pos=pos, #memory的pos
                                                                        query_pos=query_pos,
                                                                        query_pos_sub = query_pos_sub,
                                                                        query_pos_obj = query_pos_obj,
                                                                        query_sine_embed=query_sine_embed,
                                                                        query_sine_embed_sub=query_sine_embed_sub,
                                                                        query_sine_embed_obj=query_sine_embed_obj,
                                                                        so_embed = so_embed,
                                                                        is_first=(layer_id == 0))

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output) #torch.Size([300, bs, 4])
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points) #torch.Size([300, bs 4])
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points) # 第一次循环 refo_points队列长度2 [最初的ref points，第一层输出修正的ref points]
                reference_points = new_reference_points.detach()
                #最后一层的new_reference_points 没有添加到ref_points


            #===================================我加的==============================================
            #todo 原版DAB的query在做完与memory的ca之后就输出了，然后经过self.bbox_embed变换，但是这里reltr多了一层sub和entity的ca
            #todo 同样可以在后面试试哪种更合适
            tmp_sub = self.bbox_embed_sub(output_sub) #torch.Size([600, bs 4])
            tmp_sub[..., :self.query_dim] += inverse_sigmoid(reference_points_sub)
            new_reference_points_sub = tmp_sub[..., :self.query_dim].sigmoid()
            if layer_id != self.num_layers - 1:
                ref_points_sub.append(new_reference_points_sub)
            reference_points_sub = new_reference_points_sub.detach()

            tmp_obj = self.bbox_embed_obj(output_obj)
            tmp_obj[..., :self.query_dim] += inverse_sigmoid(reference_points_obj)
            new_reference_points_obj = tmp_obj[..., :self.query_dim].sigmoid()
            if layer_id != self.num_layers - 1:
                ref_points_obj.append(new_reference_points_obj)
            reference_points_obj = new_reference_points_obj.detach()

            #===================================我加的==============================================
            

            #默认True
            if self.return_intermediate:    
                intermediate.append(self.norm(output))
                intermediate_output_sub.append(self.norm_sub(output_sub))
                intermediate_output_obj.append(self.norm_obj(output_obj))

                intermediate_submaps.append(sub_maps)
                intermediate_objmaps.append(obj_maps)


        # intermediate_last = intermediate[-1]
        # output = self.norm(output)
        # flag = intermediate_last.equal(output)




        if self.norm is not None:
            output = self.norm(output) # torch.Size([300, bs, 256])
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)



        #===================================我加的==============================================
        if self.norm is not None:
            output_sub = self.norm_sub(output_sub) # torch.Size([300, bs, 256])
            if self.return_intermediate:
                intermediate_output_sub.pop()
                intermediate_output_sub.append(output_sub)

        if self.norm is not None:
            output_obj = self.norm_obj(output_obj) # torch.Size([300, bs, 256])
            if self.return_intermediate:
                intermediate_output_obj.pop()
                intermediate_output_obj.append(output_obj)

        #===================================我加的==============================================


        result = {"hs":torch.stack(intermediate).transpose(1, 2),
                  "reference":torch.stack(ref_points).transpose(1, 2),
                  "hs_sub":torch.stack(intermediate_output_sub).transpose(1, 2),
                  "reference_sub":torch.stack(ref_points_sub).transpose(1, 2),
                  "hs_obj":torch.stack(intermediate_output_obj).transpose(1, 2),
                  "reference_obj":torch.stack(ref_points_obj).transpose(1, 2),
                }


        return result, torch.stack(intermediate_submaps), torch.stack(intermediate_objmaps)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

        #===================================我加的==============================================
        self.sa_qcontent_so_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_so_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_so_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_so_proj = nn.Linear(d_model, d_model)
        self.sa_v_so_proj = nn.Linear(d_model, d_model)
        self.coupled_self_attn_so = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        self.dropout2_so = nn.Dropout(dropout)
        self.norm2_so = nn.LayerNorm(d_model)

        #? subject branch - decoupled visual attention
        self.sub_dc_ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.sub_dc_ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.sub_dc_ca_v_proj = nn.Linear(d_model, d_model)
        self.sub_dc_ca_kpos_proj = nn.Linear(d_model, d_model)
        self.sub_dc_ca_qpos_proj = nn.Linear(d_model, d_model)
        self.sub_dc_ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.sub_dc_cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)
        self.sub_dc_dropout = nn.Dropout(dropout)
        self.sub_dc_norm = nn.LayerNorm(d_model)

        #? subject branch - decoupled entity attention
        self.sub_dea_ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.sub_dea_ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.sub_dea_ca_v_proj = nn.Linear(d_model, d_model)
        self.sub_dea_cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)
        self.sub_dea_dropout = nn.Dropout(dropout)
        self.sub_dea_norm = nn.LayerNorm(d_model)
        self.sub_dea_linear1 = nn.Linear(d_model, dim_feedforward)
        self.sub_dea_dropout2 = nn.Dropout(dropout)
        self.sub_dea_linear2 = nn.Linear(dim_feedforward, d_model)
        self.sub_dea_dropout3 = nn.Dropout(dropout)
        self.sub_dea_norm3 = nn.LayerNorm(d_model)

        #* object branch - decoupled visual attention
        self.obj_dc_ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.obj_dc_ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.obj_dc_ca_v_proj = nn.Linear(d_model, d_model)
        self.obj_dc_ca_kpos_proj = nn.Linear(d_model, d_model)
        self.obj_dc_ca_qpos_proj = nn.Linear(d_model, d_model)
        self.obj_dc_ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.obj_dc_cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)
        self.obj_dc_dropout = nn.Dropout(dropout)
        self.obj_dc_norm = nn.LayerNorm(d_model)

        #* object branch - decoupled entity attention
        self.obj_dea_ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.obj_dea_ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.obj_dea_ca_v_proj = nn.Linear(d_model, d_model)
        self.obj_dea_cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)
        self.obj_dea_dropout = nn.Dropout(dropout)
        self.obj_dea_norm = nn.LayerNorm(d_model)
        self.obj_dea_linear1 = nn.Linear(d_model, dim_feedforward)
        self.obj_dea_dropout2 = nn.Dropout(dropout)
        self.obj_dea_linear2 = nn.Linear(dim_feedforward, d_model)
        self.obj_dea_dropout3 = nn.Dropout(dropout)
        self.obj_dea_norm3 = nn.LayerNorm(d_model)
        #===================================我加的==============================================



    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                      tgt_sub,
                      tgt_obj, 
                      memory,
                      tgt_mask: Optional[Tensor] = None,
                      memory_mask: Optional[Tensor] = None,
                      tgt_key_padding_mask: Optional[Tensor] = None,
                      memory_key_padding_mask: Optional[Tensor] = None,
                      pos: Optional[Tensor] = None,
                      query_pos: Optional[Tensor] = None,
                      query_pos_sub: Optional[Tensor] = None,
                      query_pos_obj: Optional[Tensor] = None,
                      query_sine_embed = None,
                      query_sine_embed_sub = None,
                      query_sine_embed_obj = None,
                      so_embed = None,
                      is_first = False):
        
        # Part 1 DABDETR
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)      # torch.Size([300, bs, 256]) target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)        # torch.Size([300, bs, 256])
            k_content = self.sa_kcontent_proj(tgt)  
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape # torch.Size([300, bs, 256])
            hw, _, _ = k_content.shape  #hw=300

            q = q_content + q_pos
            k = k_content + k_pos
            # tgt_mask   tgt_key_padding_mask  都是none
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt) # 300 bs 256
        k_content = self.ca_kcontent_proj(memory) # torch.Size([888, bs, 256])
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos) # torch.Size([888, bs, 256])  这个pos是给backbone特征图的，对应位置编码函数1

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos) #torch.Size([300, bs, 256])
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead) # torch.Size([300, bs, 8, 32])
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed) #torch.Size([300, bs, 256])
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead) # torch.Size([300, bs, 8, 32])
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2) # torch.Size([300, bs, 512])
        k = k.view(hw, bs, self.nhead, n_model//self.nhead) # torch.Size([888, bs, 8, 32])
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2) # torch.Size([888, bs, 512])

        tgt2 = self.cross_attn(    query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # Part 1 DABDETR end


        #===================================我加的==============================================

        # Part 2 subject query part
        t_num = query_pos_sub.shape[0]
        h_dim = query_pos_sub.shape[2]

        # implementation 1
        # q_content_sub = self.sa_qcontent_so_proj((tgt_sub+so_embed[0]))
        # q_pos_sub = self.sa_qpos_so_proj(query_pos_sub)
        # k_content_sub = self.sa_kcontent_so_proj((tgt_sub+so_embed[0]))
        # k_pos_sub = self.sa_kpos_so_proj(query_pos_sub)
        # v_sub = self.sa_v_so_proj((tgt_sub+so_embed[0]))

        # q_content_obj = self.sa_qcontent_so_proj((tgt_obj+so_embed[1]))
        # q_pos_obj = self.sa_qpos_so_proj(query_pos_obj)
        # k_content_obj = self.sa_kcontent_so_proj((tgt_obj+so_embed[1]))
        # k_pos_obj = self.sa_kpos_so_proj(query_pos_obj)
        # v_obj = self.sa_v_so_proj((tgt_obj+so_embed[1]))

        # q_content_so = torch.cat((q_content_sub, q_content_obj), dim=0)
        # k_content_so = torch.cat((k_content_sub, k_content_obj), dim=0)
        # q_pos_so = torch.cat((q_pos_sub, q_pos_obj), dim=0)
        # k_pos_so = torch.cat((k_pos_sub, k_pos_obj), dim=0)

        # q_so = q_content_so + q_pos_so
        # k_so = k_content_so + k_pos_so
        # v_so = torch.cat((v_sub, v_obj), dim=0)




        #这个是reltr的实现，注意tgt_sub和tgt_obj都加了相同的位置编码triplet_pos，但是在我这里先使用了不同的编码
        # q_sub = k_sub = self.with_pos_embed(self.with_pos_embed(tgt_sub, triplet_pos), so_pos[0])  #so_pos [2,256]
        # q_obj = k_obj = self.with_pos_embed(self.with_pos_embed(tgt_obj, triplet_pos), so_pos[1])


        #implementation 2
        tgt_so = torch.cat(((tgt_sub+so_embed[0]), (tgt_obj+so_embed[1])), dim=0)
        query_pos_so = torch.cat((query_pos_sub, query_pos_obj), dim=0)
        q_content_so_2 = self.sa_qcontent_so_proj(tgt_so)
        q_pos_so_2 = self.sa_qpos_so_proj(query_pos_so)
        k_content_so_2 = self.sa_kcontent_so_proj(tgt_so)
        k_pos_so_2 = self.sa_kpos_so_proj(query_pos_so)
        v_so_2 =  self.sa_v_so_proj(tgt_so)
        q_so_2 = q_content_so_2 + q_pos_so_2 # [1200, bs, 256]

        k_so_2 = k_content_so_2 + k_pos_so_2
        tgt2_so = self.coupled_self_attn_so(q_so_2, k_so_2, v_so_2)[0] # [1200, bs, 256]
        tgt_so = tgt_so + self.dropout2_so(tgt2_so) # [1200, bs, 256]

        tgt_so = self.norm2_so(tgt_so) # [1200, bs, 256]
        tgt_sub, tgt_obj = torch.split(tgt_so, t_num, dim=0)   #torch.Size([600, bs, 256]) #torch.Size([600, bs, 256])


        # # 虽然他们的不完全一样，但是他们直接的误差应该是浮点数造成的，而且非常的小，两种实现都是对的
        # tolerance = 1e-6  # Adjust as needed
        # are_close1 = torch.allclose(q_pos_so, q_pos_so_2, atol=tolerance)
        # print("Are q_pos_so and q_pos_so_2 close within tolerance?", are_close1)
        # are_close2 = torch.allclose(q_content_so, q_content_so_2, atol=tolerance)
        # are_close3 = torch.allclose(k_content_so, k_content_so_2, atol=tolerance)
        # are_close4 = torch.allclose(v_so, v_so_2, atol=tolerance)
        # print(are_close2, are_close3, are_close4) # True True True


        # subject branch - decoupled visual attention  这一块负责sub query和memory做cross attn
        q_content_sub_dc = self.sub_dc_ca_qcontent_proj(tgt_sub) # torch.Size([600, bs, 256])
        k_content_sub_dc = self.sub_dc_ca_kcontent_proj(memory)  # torch.Size([1000, bs, 256])  
        k_pos_sub_dc = self.sub_dc_ca_kpos_proj(pos) # torch.Size([1000, bs, 256])  这个pos是给backbone特征图的
        v_sub_dc = self.sub_dc_ca_v_proj(memory)

        num_queries_sub, bs, n_model = q_content_sub_dc.shape # 600 bs 256
        hw_sub_dc, _, _ = k_content_sub_dc.shape

        if is_first or self.keep_query_pos:
            q_pos_sub_dc = self.sub_dc_ca_qpos_proj(query_pos_sub) # torch.Size([600, bs, 256])
            q_sub_dc = q_content_sub_dc + q_pos_sub_dc # torch.Size([600, bs, 256])
            k_sub_dc = k_content_sub_dc + k_pos_sub_dc # # torch.Size([1000, bs, 256])
        else:
            q_sub_dc = q_content_sub_dc 
            k_sub_dc = k_content_sub_dc 
        
        q_sub_dc = rearrange(q_sub_dc, "n bs (head head_dim) -> n bs head head_dim", head=self.nhead, head_dim=(n_model//self.nhead))
        query_sine_embed_sub_dc = self.sub_dc_ca_qpos_sine_proj(query_sine_embed_sub)
        query_sine_embed_sub_dc = rearrange(query_sine_embed_sub_dc, "n bs (head head_dim) -> n bs head head_dim", head=self.nhead, head_dim=(n_model//self.nhead))
        q_sub_dc = torch.cat([q_sub_dc, query_sine_embed_sub_dc], dim=3).view(num_queries_sub, bs, n_model * 2)

        k_sub_dc = rearrange(k_sub_dc, "hw bs (head head_dim) -> hw bs head head_dim", head=self.nhead, head_dim=(n_model//self.nhead))
        k_pos_sub_dc = rearrange(k_pos_sub_dc, "hw bs (head head_dim) -> hw bs head head_dim", head=self.nhead, head_dim=(n_model//self.nhead))
        k_sub_dc = torch.cat([k_sub_dc, k_pos_sub_dc], dim=3).view(hw_sub_dc, bs, n_model * 2)

        # [600, bs, 256]      [bs, 600, 1000] 
        tgt2_sub_dc, sub_maps = self.sub_dc_cross_attn( query=q_sub_dc,
                                                        key=k_sub_dc,
                                                        value=v_sub_dc, 
                                                        attn_mask=memory_mask, #None
                                                        key_padding_mask=memory_key_padding_mask)

        tgt_sub_dc = tgt_sub + self.sub_dc_dropout(tgt2_sub_dc) # torch.Size([600, bs, 256])
        tgt_sub_dc = self.sub_dc_norm(tgt_sub_dc) # torch.Size([600, bs, 256])

        # * subject branch - decoupled entity attention 这块负责tgt_sub_dc再和tgt做ca，tgt就是DAB原版ca的输出过ffn
        # query: tgt_sub_dc   key: tgt
        # todo query_pos: 方案1：使用692行的query_sine_embed_sub_dc  方案2：self.sub_dea_ca_qpos_sine_proj(query_sine_embed_sub)
        # todo key_pos: 方案1：使用592行的query_sine_embed           方案2：self.sub_dea_ca_kpos_sine_proj(query_sine_embed)
        # todo   这里使用两个方案1节省参数           后续可以实验方案2

        q_content_sub_dea = self.sub_dea_ca_qcontent_proj(tgt_sub_dc) # torch.Size([600, bs, 256])
        k_content_sub_dea = self.sub_dea_ca_kcontent_proj(tgt) # torch.Size([300, bs, 256])
        q_pos_sub_dea = query_sine_embed_sub_dc #[600, bs, 8, 32]
        k_pos_sub_dea = query_sine_embed #[300, bs, 8, 32]

        q_content_sub_dea = rearrange(q_content_sub_dea, 
                                      "n bs (head head_dim) -> n bs head head_dim", 
                                      head=self.nhead, head_dim=(n_model//self.nhead))
        # q_sub_dea torch.Size([600, bs, 512])
        q_sub_dea = torch.cat([q_content_sub_dea, q_pos_sub_dea], dim=3).view(num_queries_sub, bs, n_model * 2)

        k_content_sub_dea = rearrange(k_content_sub_dea, 
                                      "n bs (head head_dim) -> n bs head head_dim", 
                                      head=self.nhead, head_dim=(n_model//self.nhead))
        # k_sub_dea torch.Size([300, bs, 512])
        k_sub_dea = torch.cat([k_content_sub_dea, k_pos_sub_dea], dim=3).view(num_queries, bs, n_model * 2)

        v_sub_dea = self.sub_dea_ca_v_proj(tgt) # torch.Size([300, bs, 256])

        # torch.Size([600, bs, 256])
        tgt2_sub_dea = self.sub_dea_cross_attn( query=q_sub_dea,
                                                key=k_sub_dea,
                                                value=v_sub_dea,
                                                attn_mask=None,
                                                key_padding_mask=None
                                               )[0]
        
        tgt_sub_dea = tgt_sub_dc + self.sub_dea_dropout2(tgt2_sub_dea)
        tgt_sub_dea = self.sub_dea_norm(tgt_sub_dea)
        tgt_sub_dea2 = self.sub_dea_linear2(self.sub_dea_dropout(self.activation(self.sub_dea_linear1(tgt_sub_dea))))
        tgt_sub_dea = tgt_sub_dea + self.sub_dea_dropout3(tgt_sub_dea2)
        tgt_sub_dea = self.sub_dea_norm3(tgt_sub_dea)

        # Part 3 object query part
        # * object branch - decoupled visual attention  这一块负责obj query和memory做cross attn
        q_content_obj_dc = self.obj_dc_ca_qcontent_proj(tgt_obj) # torch.Size([600, bs, 256])
        k_content_obj_dc = self.obj_dc_ca_kcontent_proj(memory)  # torch.Size([1000, bs, 256])  
        k_pos_obj_dc = self.obj_dc_ca_kpos_proj(pos) # torch.Size([1000, bs, 256])  这个pos是给backbone特征图的
        v_obj_dc = self.obj_dc_ca_v_proj(memory)

        if is_first or self.keep_query_pos:
            q_pos_obj_dc = self.obj_dc_ca_qpos_proj(query_pos_obj) # torch.Size([600, bs, 256])
            q_obj_dc = q_content_obj_dc + q_pos_obj_dc # torch.Size([600, bs, 256])
            k_obj_dc = k_content_obj_dc + k_pos_obj_dc # # torch.Size([1000, bs, 256])
        else:
            q_obj_dc = q_content_obj_dc 
            k_obj_dc = k_content_obj_dc 

        q_obj_dc = rearrange(q_obj_dc, "n bs (head head_dim) -> n bs head head_dim", head=self.nhead, head_dim=(n_model//self.nhead))
        query_sine_embed_obj_dc = self.obj_dc_ca_qpos_sine_proj(query_sine_embed_obj)
        query_sine_embed_obj_dc = rearrange(query_sine_embed_obj_dc, "n bs (head head_dim) -> n bs head head_dim", head=self.nhead, head_dim=(n_model//self.nhead))
        q_obj_dc = torch.cat([q_obj_dc, query_sine_embed_obj_dc], dim=3).view(num_queries_sub, bs, n_model * 2)

        k_obj_dc = rearrange(k_obj_dc, "hw bs (head head_dim) -> hw bs head head_dim", head=self.nhead, head_dim=(n_model//self.nhead))
        k_pos_obj_dc = rearrange(k_pos_obj_dc, "hw bs (head head_dim) -> hw bs head head_dim", head=self.nhead, head_dim=(n_model//self.nhead))
        k_obj_dc = torch.cat([k_obj_dc, k_pos_obj_dc], dim=3).view(hw_sub_dc, bs, n_model * 2)

        tgt2_obj_dc, obj_maps = self.obj_dc_cross_attn(  query=q_obj_dc,
                                                         key=k_obj_dc,
                                                         value=v_obj_dc, 
                                                         attn_mask=memory_mask, #None
                                                         key_padding_mask=memory_key_padding_mask)

        tgt_obj_dc = tgt_obj + self.obj_dc_dropout(tgt2_obj_dc) # torch.Size([600, bs, 256])
        tgt_obj_dc = self.obj_dc_norm(tgt_obj_dc) # torch.Size([600, bs, 256])


        # * object branch - decoupled entity attention 这块负责tgt_obj_dc再和tgt做ca，tgt就是DAB原版ca的输出过ffn
        q_content_obj_dea = self.obj_dea_ca_qcontent_proj(tgt_obj_dc) # torch.Size([600, bs, 256])
        k_content_obj_dea = self.obj_dea_ca_kcontent_proj(tgt) # torch.Size([300, bs, 256])
        q_pos_obj_dea = query_sine_embed_obj_dc #[600, bs, 8, 32]
        k_pos_obj_dea = query_sine_embed #[300, bs, 8, 32]

        q_content_obj_dea = rearrange(q_content_obj_dea, 
                                      "n bs (head head_dim) -> n bs head head_dim", 
                                      head=self.nhead, head_dim=(n_model//self.nhead))
        # q_obj_dea torch.Size([600, bs, 512])
        q_obj_dea = torch.cat([q_content_obj_dea, q_pos_obj_dea], dim=3).view(num_queries_sub, bs, n_model * 2)

        k_content_obj_dea = rearrange(k_content_obj_dea, 
                                      "n bs (head head_dim) -> n bs head head_dim", 
                                      head=self.nhead, head_dim=(n_model//self.nhead))
        # k_obj_dea torch.Size([300, bs, 512])
        k_obj_dea = torch.cat([k_content_obj_dea, k_pos_obj_dea], dim=3).view(num_queries, bs, n_model * 2)
        v_obj_dea = self.obj_dea_ca_v_proj(tgt) # torch.Size([300, bs, 256])

        # torch.Size([600, bs, 256])
        tgt2_obj_dea = self.obj_dea_cross_attn( query=q_obj_dea,
                                                key=k_obj_dea,
                                                value=v_obj_dea,
                                                attn_mask=None,
                                                key_padding_mask=None
                                               )[0]
        
        tgt_obj_dea = tgt_obj_dc + self.obj_dea_dropout2(tgt2_obj_dea)
        tgt_obj_dea = self.obj_dea_norm(tgt_obj_dea)
        tgt_obj_dea2 = self.obj_dea_linear2(self.obj_dea_dropout(self.activation(self.obj_dea_linear1(tgt_obj_dea))))
        tgt_obj_dea = tgt_obj_dea + self.obj_dea_dropout3(tgt_obj_dea2)
        tgt_obj_dea = self.obj_dea_norm3(tgt_obj_dea)
        #===================================我加的==============================================

        #tgt_triplet = torch.cat((tgt_sub_dea, tgt_obj_dea), dim=-1) #torch.Size([600, bs, 512])
        return tgt, tgt_sub_dea, tgt_obj_dea, sub_maps, obj_maps
    



def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_entities,           # DAB独有 其他都一样  args.num_queries
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=4,   #DAB独有
        activation=args.transformer_activation, #DAB独有
        num_patterns=args.num_patterns, #DAB独有
    )



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")