from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import dropout_, nn
import warnings
import math
from torch._C import _infer_size, _add_docstr
import einops

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, need_patch=False):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        y = self.attnpool(x)

        return x, y


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


linear = _add_docstr(
    torch._C._nn.linear,
    r"""
linear(input, weight, bias=None) -> Tensor
Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
Shape:
    - Input: :math:`(*, in\_features)` where `*` means any number of
      additional dimensions, including none
    - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
    - Bias: :math:`(out\_features)` or :math:`()`
    - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
""")

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, step: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.step = step

    def attention(self, x: torch.Tensor, detection_attn_mask=None, box_coords=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if detection_attn_mask is not None:
            
            
            query = x
            key = x
            value = x
            embed_dim_to_check = self.attn.embed_dim
            num_heads = self.attn.num_heads
            in_proj_weight = self.attn.in_proj_weight
            in_proj_bias = self.attn.in_proj_bias
            bias_k = self.attn.bias_k
            bias_v = self.attn.bias_v
            static_k = None
            static_v = None
            add_zero_attn = self.attn.add_zero_attn
            dropout_p = self.attn.dropout
            out_proj_weight = self.attn.out_proj.weight
            out_proj_bias = self.attn.out_proj.bias
            training = self.attn.training
            key_padding_mask = None
            need_weights = False
            average_attn_weights = True
            attn_mask=detection_attn_mask.to(dtype=x.dtype, device=x.device)
            is_batched = True
            tgt_len, bsz, embed_dim = query.shape
            src_len, _, _ = key.shape
            expand_bsz = detection_attn_mask.shape[0]//num_heads
            
            assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"

            if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
                head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
            else:
                head_dim = embed_dim // num_heads
            assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

            assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
            
            q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
            
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            
            ### SKIP attn_mask shape check
            
            if bias_k is not None and bias_v is not None:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = F._pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = F._pad(key_padding_mask, (0, 1))
            else:
                assert bias_k is None
                assert bias_v is None
            
            q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)[:, :1, :]
            if static_k is None:
                k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            else:
                # TODO finish disentangling control flow so we don't do in-projections when statics are passed
                assert static_k.size(0) == bsz * num_heads, \
                    f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
                assert static_k.size(2) == head_dim, \
                    f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
                k = static_k
            if static_v is None:
                v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            else:
                # TODO finish disentangling control flow so we don't do in-projections when statics are passed
                assert static_v.size(0) == bsz * num_heads, \
                    f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
                assert static_v.size(2) == head_dim, \
                    f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
                v = static_v
            
            src_len = k.size(1)
            if not training:
                dropout_p = 0.0
            
            B, Nt, E = q.shape
            B, Ns, E = k.shape
            q = q / math.sqrt(E)
            attn = torch.bmm(q, k.transpose(-2, -1))
            if box_coords[0].dim() == 2:
                    box_len_per_batch = list(map(lambda x: len(x), box_coords))
            elif box_coords[0].dim() == 3:
                assert len(box_coords) == 1
                num_heads = self.attn.num_heads
                box_len_per_batch = [len(detection_attn_mask) // num_heads]
            else:
                assert False

            attn_expand_list = []
            v_expand_list = []
            for idx, box_len in enumerate(box_len_per_batch):
                attn_expand_list.append(attn.reshape(bsz, num_heads, Nt, Ns)[idx].repeat(box_len, 1, 1))
                v_expand_list.append(v.reshape(bsz, num_heads, Ns, E)[idx].repeat(box_len, 1, 1))
            attn_expand = torch.cat(attn_expand_list, dim=0)
            v_expand = torch.cat(v_expand_list, dim=0)
            attn_expand += attn_mask
            attn_expand = F.softmax(attn_expand, dim=-1)

            if dropout_p > 0.0:
                attn_expand = F.dropout(attn_expand, p=dropout_p)

            attn_output_expand = torch.bmm(attn_expand, v_expand)

            attn = F.softmax(attn, dim=-1)
            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p)
            attn_output = torch.bmm(attn, v)
            
            attn_output = attn_output.transpose(0, 1).contiguous().view(bsz, embed_dim)
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(1, bsz, attn_output.size(1))
            
            attn_output_expand = attn_output_expand.transpose(0, 1).contiguous().view(expand_bsz, embed_dim)
            attn_output_expand = linear(attn_output_expand, out_proj_weight, out_proj_bias)
            attn_output_expand = attn_output_expand.view(1, expand_bsz, attn_output_expand.size(1))
            
            return attn_output, attn_output_expand

        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, seq_dict):
        x = seq_dict['x']
        detection_attn_mask = seq_dict['detection_attn_mask']
        layers = seq_dict['layers']
        box_coords = seq_dict['box_coords']
        need_patch = seq_dict['need_patch']
        if detection_attn_mask is not None:
            if self.step != layers -1:
                x = x + self.attention(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))
                return {'x':x, 'detection_attn_mask': detection_attn_mask, 'layers': layers, 'box_coords': box_coords, 'need_patch': need_patch}
            else:

                if need_patch:
                    global_x = x + self.attention(self.ln_1(x))
                    global_x = global_x + self.mlp(self.ln_2(global_x))
                
                cls, mask_cls = self.attention(self.ln_1(x), detection_attn_mask, box_coords)
                cls = x[:1, :, :] + cls
                cls = cls + self.mlp(self.ln_2(cls))
                
                x_expand_list = []
                if box_coords[0].dim() == 2:
                    box_len_per_batch = list(map(lambda x: len(x), box_coords))
                elif box_coords[0].dim() == 3:
                    assert len(box_coords) == 1
                    num_heads = self.attn.num_heads
                    box_len_per_batch = [len(detection_attn_mask) // num_heads]
                else:
                    assert False
                for idx, box_len in enumerate(box_len_per_batch):
                    x_expand_list.append(x[:,idx].unsqueeze(1).repeat(1, box_len, 1))
                x_expand = torch.cat(x_expand_list, dim=1)
                mask_cls = x_expand[:1, :, :] + mask_cls
                mask_cls = mask_cls + self.mlp(self.ln_2(mask_cls))
                


                if need_patch:
                    return {'x':mask_cls, 'cls': cls, 'global_x': global_x, 'detection_attn_mask': detection_attn_mask, 'layers': layers, 'box_coords': box_coords}
                else:
                    return {'x':mask_cls, 'cls': cls, 'detection_attn_mask': detection_attn_mask, 'layers': layers, 'box_coords': box_coords}
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return {'x':x, 'detection_attn_mask': None, 'layers': None, 'box_coords': None, 'need_patch': need_patch}


def torch_intersection(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def find_det_mask(box_coords, L, patch_size, heads, pose_attention_weight=None):
    
    if box_coords[0].shape[1] == 4:
        box_coords = torch.cat(box_coords, dim=0)
        batch_size = box_coords.shape[0]
        det_mask = torch.zeros([batch_size, 1, L]).to(box_coords.device)
        det_mask[:, 0, 0] = 1
        width = int(L**0.5)
        box_coords_normed = box_coords/patch_size
        box_coords_int = torch.cat([torch.floor(box_coords_normed[:, :2]), torch.ceil(box_coords_normed[:, 2:])], dim=1).clip(0, width)
        box_coords_area_wh = 1 - torch.abs(box_coords_int - box_coords_normed) 
        for b_i, (box_coord_area_wh, box_coord_int) in enumerate(zip(box_coords_area_wh, box_coords_int)):  
            a = int(box_coord_int[0])
            b = int(box_coord_int[1])
            c = int(box_coord_int[2])
            d = int(box_coord_int[3])
            x, y, z, w = box_coord_area_wh.unsqueeze(1)
            
            row = torch.arange(width*b+a+1, width*b+c+1).to(x.device)
            mask_index = row.repeat(d-b) + torch.arange(d-b).to(x.device).repeat_interleave(c-a) * width
            
            if c-a == 1 and d-b == 1:
                mask_area = (x+z-1) * (y+w-1)
            elif c-a == 1:
                mask_area = (x+z-1) * torch.cat([y, torch.ones(d-b-2).to(x.device), w])
            elif d-b == 1:
                mask_area = (y+w-1) * torch.cat([x, torch.ones(c-a-2).to(x.device), z])
            else:
                area_row = torch.cat([x, torch.ones(c-a-2).to(x.device), z])
                area_column = torch.cat([y, torch.ones(d-b-2).to(x.device), w])
                mask_area = area_row.unsqueeze(0) * area_column.unsqueeze(1)
            
            det_mask[b_i, 0, mask_index] = mask_area.view(-1)
        

        return torch.ceil(det_mask)     




def find_mask_idx(box_coord_int, width):
    a = int(box_coord_int[0])
    b = int(box_coord_int[1])
    c = int(box_coord_int[2])
    d = int(box_coord_int[3]) 
    row = torch.arange(width*b+a+1, width*b+c+1)  
    mask_index = row.repeat(d-b) + torch.arange(d-b).repeat_interleave(c-a) *width
    
    return mask_index

def fine_det_center_point(box_coords, input_resolution, patch_size):
    with torch.no_grad():
        box_coords = torch.cat(box_coords, dim=0)
        width = input_resolution // patch_size
        batch_size = box_coords.shape[0]
        box_coords_center = (box_coords[:, :2] + box_coords[:, 2:]) / 2
        box_coords_normed = box_coords_center/patch_size
        box_coords_int = torch.floor(box_coords_normed).clip(0, width)
        center_patch_index = width * box_coords_int[:, 1] + box_coords_int[:, 0] + 1

        return center_patch_index.long()


def find_det_mask_with_area(box_coords, L, patch_size, heads, pose_attention_weight=None):
    
    if box_coords[0].shape[1] == 4:
        box_coords = torch.cat(box_coords, dim=0)
        batch_size = box_coords.shape[0]
        det_mask = torch.zeros([batch_size, 1, L]).to(box_coords.device)
        det_mask[:, 0, 0] = 1
        width = int(L**0.5)
        box_coords_normed = box_coords/patch_size
        box_coords_int = torch.cat([torch.floor(box_coords_normed[:, :2]), torch.ceil(box_coords_normed[:, 2:])], dim=1).clip(0, width)
        box_coords_area_wh = 1 - torch.abs(box_coords_int - box_coords_normed) 
        for b_i, (box_coord_area_wh, box_coord_int) in enumerate(zip(box_coords_area_wh, box_coords_int)):  
            a = int(box_coord_int[0])
            b = int(box_coord_int[1])
            c = int(box_coord_int[2])
            d = int(box_coord_int[3])
            x, y, z, w = box_coord_area_wh.unsqueeze(1)
            
            row = torch.arange(width*b+a+1, width*b+c+1).to(x.device)
            mask_index = row.repeat(d-b) + torch.arange(d-b).to(x.device).repeat_interleave(c-a) * width
            
            if c-a == 1 and d-b == 1:
                mask_area = (x+z-1) * (y+w-1)
            elif c-a == 1:
                mask_area = (x+z-1) * torch.cat([y, torch.ones(d-b-2).to(x.device), w])
            elif d-b == 1:
                mask_area = (y+w-1) * torch.cat([x, torch.ones(c-a-2).to(x.device), z])
            else:
                area_row = torch.cat([x, torch.ones(c-a-2).to(x.device), z])
                area_column = torch.cat([y, torch.ones(d-b-2).to(x.device), w])
                mask_area = area_row.unsqueeze(0) * area_column.unsqueeze(1)
            
            det_mask[b_i, 0, mask_index] = mask_area.view(-1)
                
        return det_mask     
    
    elif box_coords[0].shape[1] == 8:
        print("DEFR?")
        box_coords_h = list(map(lambda x: x[:, :4], box_coords))
        box_coords_o = list(map(lambda x: x[:, 4:], box_coords))
        
        det_mask_h = find_det_mask_with_area(box_coords_h, L, patch_size, heads)
        det_mask_o = find_det_mask_with_area(box_coords_o, L, patch_size, heads)
        
        return (det_mask_h + det_mask_o).clip(0,1)
    elif box_coords[0].shape[1] == 672/patch_size: ### Segmentation case
        print("segmentation clip using")
        box_segments = torch.cat(box_coords, dim=0)
        batch_size = box_segments.shape[0]
        det_mask = torch.zeros([batch_size, 1, L]).to(box_segments.device)
        det_mask[:, 0, 0] = 1
        width = int(L**0.5)
        seg_mask = box_segments
        #seg_mask = F.interpolate(box_segments.unsqueeze(0), size=[width, width], mode='bilinear')[0]
        det_mask[:, 0, 1:] = seg_mask.view(batch_size, -1)
        
        return det_mask
    
    elif box_coords[0].shape[1] == 17:  ### pose box case

        assert len(box_coords) == 1
        box_coords = box_coords[0]
        pose_box_coords = list(map(lambda x: x.squeeze(0), torch.chunk(box_coords, box_coords.shape[0], dim=0)))
        pose_det_mask = find_det_mask_with_area(pose_box_coords, L, patch_size, heads)
        pose_det_mask = pose_det_mask.reshape(-1, 17, 1, L)
        n_h = pose_det_mask.shape[0]
        total_n = pose_attention_weight.shape[0]
        assert total_n % n_h == 0
        n = total_n // n_h
        pose_det_mask = pose_det_mask.repeat_interleave(n, dim=0)
        pose_det_mask = torch.sum(pose_det_mask * pose_attention_weight.unsqueeze(-1).unsqueeze(-1), dim=1).clip(1e-9, )

        return pose_det_mask
        
        
    else:
        assert False 

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.attn_mask = attn_mask
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, i, attn_mask) for i in range(layers)])
        
    def forward(self, x: torch.Tensor, patch_size=None, box_coords=None, pose_attention_weight=None, need_patch=False):
        #return self.resblocks(x)
        L, B, C = x.shape
        #print("x shape:", x.shape)
        if box_coords is not None:
            detection_attn_mask = find_det_mask_with_area(box_coords, L, patch_size, self.heads, pose_attention_weight)
            detection_attn_mask = torch.log(detection_attn_mask).repeat_interleave(self.heads, dim=0)
            seq_dict = {'x':x, 'detection_attn_mask': detection_attn_mask, 'layers': self.layers, 'box_coords': box_coords, 'need_patch': need_patch}
        else:
            seq_dict = {'x':x, 'detection_attn_mask': None, 'layers': None, 'box_coords': None, 'need_patch': need_patch}
        return self.resblocks(seq_dict)

    
        
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, box_coords=None, box_segs=None, pose_attention_weight=None, need_patch=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        
        if box_coords is not None :
            if box_coords[0].shape[1] == 4 or box_coords[0].shape[1] == 17:
                if box_segs is None:
                    res = self.transformer(x, self.patch_size, box_coords, pose_attention_weight, need_patch)
                else:
                    res = self.transformer(x, self.patch_size, box_segs, pose_attention_weight, need_patch)
                mask_cls_tok = res['x']
                cls_tok = res['cls']
                if need_patch:
                    patch_tok = res['global_x']
                    patch_tok = patch_tok.permute(1, 0, 2)[:, 1:, :]
                
                mask_cls_tok = self.ln_post(mask_cls_tok[0])
                cls_tok = self.ln_post(cls_tok[0])
                
                if need_patch:
                    return mask_cls_tok, cls_tok, patch_tok
                else:
                    return mask_cls_tok, cls_tok

            else:
                assert False

        else:
            x = self.transformer(x, need_patch=need_patch)['x']
            
            x = x.permute(1, 0, 2)  # LND -> NLD

            cls_tok = self.ln_post(x[:, 0, :])
            img_tok = x[:,1:,:]
            return cls_tok, img_tok


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, box_coords=None, box_segs=None, pose_attention_weight=None, need_patch=False):
        if box_coords is None:
            return self.visual(image.type(self.dtype), need_patch=need_patch)
        else:
            if box_segs is None:
                if pose_attention_weight is None:
                    return self.visual(image.type(self.dtype), box_coords=box_coords, need_patch=need_patch)
                else:
                    return self.visual(image.type(self.dtype), box_coords=box_coords, pose_attention_weight=pose_attention_weight, need_patch=need_patch)
            else:
                return self.visual(image.type(self.dtype), box_coords=box_coords, box_segs=box_segs, need_patch=need_patch)    
    

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)['x']
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
