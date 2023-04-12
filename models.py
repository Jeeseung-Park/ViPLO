import torch

import PIL.Image as im

from torchvision.transforms import Compose, ToTensor, ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = im.BICUBIC

from collections import OrderedDict
from torch import nn, Tensor

from typing import Optional, List
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import transform
import torch.nn.functional as F

from ops import warp_affine_joints, transform_preds, warpaffine_image
import numpy as np

import pocket.models as models
import clip

from transforms import HOINetworkTransform
from interaction_head import InteractionHead, GraphHead

class GenericHOINetwork(nn.Module):
    """A generic architecture for HOI classification
torchvision Imagelist
    -----------
        backbone: nn.Module
        interaction_head: nn.Module
        transform: nn.Module
        postprocess: bool
            If True, rescale bounding boxes to original image size
    """
    def __init__(self,
        backbone: nn.Module, backbone_name: str, interaction_head: nn.Module,
        transform: nn.Module, postprocess: bool = True, rank: int = 0, patch_size: int = 32, human_idx: int = 0, pose : bool = True
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.interaction_head = interaction_head
        self.transform = transform
        self.postprocess = postprocess
        self.rank = rank
        self.patch_size = patch_size 
        self.human_idx = human_idx
        self.pose = pose
        self.topilimage = ToPILImage()
        self.instance_norm = nn.InstanceNorm2d(256, affine=False)
        

    def preprocess(self,
        images: List[Tensor],
        detections: List[dict],
        targets: Optional[List[dict]] = None
    ):
        device = torch.device(f"cuda:{self.rank}")
        original_image_sizes = [img.shape[-2:] for img in images]
        if self.backbone_name == "resnet50":
            images, targets = self.transform(images, targets)
            
            for det, o_im_s, im_s in zip(
           detections, original_image_sizes, images.image_sizes
        ):
                boxes = det['boxes']
                boxes = transform.resize_boxes(boxes, o_im_s, im_s)
                det['boxes'] = boxes
                
            return images, detections, targets, original_image_sizes, [None for _ in range(len(detections))]
                
        elif self.backbone_name == "CLIP" or self.backbone_name == "CLIP_CLS":
            topilimage = ToPILImage()
            totensor = ToTensor()
            processed_image_list = []
            trans_list = []
            img_meta_list = []
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(-1, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(-1, 1, 1)
            for img in images:
                img = topilimage(img)
                img, trans, img_meta = warpaffine_image(img, n_px=672, device=device)
                img = totensor(img).to(device)
                processed_image_list.append(img.sub_(mean).div_(std))
                trans_list.append(trans)
                img_meta_list.append(img_meta)
            
            processed_image_sizes = [img.shape[-2:] for img in processed_image_list]
            if targets is None:
                for det, o_im_s, im_s, trans in zip(
        detections, original_image_sizes, processed_image_sizes, trans_list
        ):          
                    boxes = det['boxes']
                    boxes = warp_affine_joints(boxes, trans)
                    det['boxes'] = boxes

                    if self.pose:
                        human_joints = det['human_joints']
                        human_joints = warp_affine_joints(human_joints, trans)
                        det['human_joints'] = human_joints    
                        # boxes = det['boxes']
                        # human_joints_score = det['human_joints_score']
                        # boxes_label = det['labels']
                        # human_boxes = boxes[boxes_label == self.human_idx]

            
            else:
            
                for det, tar, o_im_s, im_s, trans in zip(
            detections, targets, original_image_sizes, processed_image_sizes, trans_list
            ):      
                    target_h = tar['boxes_h']
                    target_h = warp_affine_joints(target_h, trans)
                    tar['boxes_h'] = target_h    
                    target_o = tar['boxes_o']
                    target_o = warp_affine_joints(target_o, trans)
                    tar['boxes_o'] = target_o   
                    boxes = det['boxes']
                    boxes = warp_affine_joints(boxes, trans)
                    det['boxes'] = boxes

                    if self.pose:
                        human_joints = det['human_joints']
                        human_joints = warp_affine_joints(human_joints, trans)
                        det['human_joints'] = human_joints    

                        # human_joints_score = det['human_joints_score']
                        # boxes = det['boxes']
                        # boxes_label = det['labels']
                        # human_boxes = boxes[boxes_label == self.human_idx]


                        tar_human_joints = tar['human_joints']
                        tar_human_joints = warp_affine_joints(tar_human_joints, trans)
                        tar['human_joints'] = tar_human_joints
                        
                        #boxes_o = tar['boxes_o'][tar['object'] == self.human_idx]

                        
                        
            return torch.stack(processed_image_list, dim=0), detections, targets, original_image_sizes, img_meta_list
            
           
        else: 
            raise ValueError("Not supported backbone name")

    def forward(self,
        images: List[Tensor],
        detections: List[dict],
        targets: Optional[List[dict]] = None
    ) :
        """
        Parameters:
        -----------
            images: List[Tensor]
            detections: List[dict]
            targets: List[dict]
        Returns:
        --------
            results: List[dict]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images, detections, targets, original_image_sizes, img_metas = self.preprocess(
                images, detections, targets)
        
        if self.backbone_name == "resnet50":
            features = self.backbone(images.tensors)
            results = self.interaction_head(features, detections, 
            images.image_sizes, targets)
           
        elif self.backbone_name == "CLIP_CLS":
            image_sizes = [img.shape[-2:] for img in images]
            features = OrderedDict()
            results = self.interaction_head(features, detections, image_sizes, targets, images)
        
        elif self.backbone_name == "CLIP": 
            image_sizes = [img.shape[-2:] for img in images]
            global_features, patch_features = self.backbone.encode_image(images)
            features = OrderedDict()
            B, L, C = patch_features.shape
            features['0'] = patch_features.permute(0,2,1).reshape(B, C, int(L**0.5), int(L**0.5)).to(dtype=torch.float32)
            features['3'] = global_features.to(dtype=torch.float32)
            results = self.interaction_head(features, detections, image_sizes, targets)
        else:
            raise ValueError("Not supported backbone name")
            
        if self.postprocess and results is not None:
            if self.backbone_name == 'resnet50':
                return self.transform.postprocess(
                    results,
                    images.image_sizes,
                    original_image_sizes
                )
            elif self.backbone_name == 'CLIP' or self.backbone_name == "CLIP_CLS":
                if self.training:
                    loss = results.pop()
                for pred, im_s, o_im_s, img_meta in zip(results, image_sizes, original_image_sizes, img_metas):
                    boxes_h, boxes_o = pred['boxes_h'], pred['boxes_o']
                    center = img_meta['center']
                    n_px = img_meta['n_px']
                    scale = img_meta['scale']
                    boxes_h = transform_preds(boxes_h.reshape(-1, 2), center, scale, [n_px, n_px], use_udp=True).reshape(-1, 4)
                    boxes_o = transform_preds(boxes_o.reshape(-1, 2), center, scale, [n_px, n_px], use_udp=True).reshape(-1, 4)
                    pred['boxes_h'], pred['boxes_o'] = boxes_h, boxes_o

                if self.training:
                    results.append(loss)
                return results
            else:
                raise ValueError("Not supported backbone name")
        else:
            return results

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class VIPLO(GenericHOINetwork):
    def __init__(self,
        object_to_action: List[list],
        object_n_verb_to_interaction,
        object_to_interaction,
        verb_list,
        human_idx: int,
        # Backbone parameters
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        # Pooler parameters
        output_size: int = 7,
        sampling_ratio: int = 2,
        # Box pair head parameters
        node_encoding_size: int = 1024,
        representation_size: int = 1024,
        num_classes: int = 117,
        box_score_thresh: float = 0.2,
        fg_iou_thresh: float = 0.5,
        num_iterations: int = 2,
        distributed: bool = False,
        # Transformation parameters
        min_size: int = 800, max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        postprocess: bool = True,
        # Preprocessing parameters
        box_nms_thresh: float = 0.5,
        max_human: int = 15,
        max_object: int = 15,
        rank: int = 0,
        patch_size: int = 32,
        pose: bool = True,
    ):
        
        if backbone_name == "resnet50":
            detector = models.fasterrcnn_resnet_fpn(backbone_name,
                pretrained=pretrained)
            backbone = detector.backbone
            out_channels = backbone.out_channels
            logit_scale = None         
            
            
        elif backbone_name == "CLIP" or backbone_name == "CLIP_CLS":
            backbone, _ = clip.load(f"ViT-B/{patch_size}", jit=False)
            del backbone.token_embedding
            del backbone.positional_embedding
            del backbone.ln_final
            del backbone.text_projection
            del backbone.transformer
            del backbone.vocab_size
            del backbone.visual.proj

            logit_scale =  None
            del backbone.logit_scale
            pretrained_img_size = 224
            input_img_size = 672
            scale_factor = input_img_size // pretrained_img_size
            pretrained_width = pretrained_img_size // patch_size
            input_width = scale_factor * pretrained_width
            out_channels = 768
            backbone = backbone.float()
            cls_pos_embedding = backbone.visual.positional_embedding[:1]
            pre_pos_embedding = backbone.visual.positional_embedding[1:].view(pretrained_width,pretrained_width,-1).permute(2,0,1)
            
            post_pos_embedding = F.interpolate(pre_pos_embedding.unsqueeze(0), scale_factor=scale_factor, mode='bilinear')[0]
            expanded_pos_embedding = torch.cat([cls_pos_embedding, post_pos_embedding.permute(1,2,0).view(input_width*input_width,-1)], dim=0)
            backbone.visual.positional_embedding = torch.nn.Parameter(expanded_pos_embedding)
        else:
            raise ValueError("Not supported backbone name")
        
        if backbone_name == "resnet50":
            box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=output_size,
            sampling_ratio=sampling_ratio
        )
        elif backbone_name == "CLIP":
            box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=output_size, sampling_ratio=sampling_ratio)
        elif backbone_name == "CLIP_CLS":
            box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=5, sampling_ratio=sampling_ratio)
        else:
            raise ValueError("Not supported backbone name")
        
  
        box_pair_head = GraphHead(
            verb_list = verb_list,
            backbone=backbone,
            out_channels=out_channels,
            roi_pool_size=output_size,
            node_encoding_size=node_encoding_size,
            representation_size=representation_size,
            num_cls=num_classes,
            human_idx=human_idx,
            object_class_to_target_class=object_to_action,
            object_class_to_interaction_class=object_to_interaction,
            fg_iou_thresh=fg_iou_thresh,
            num_iter=num_iterations,
            backbone_name=backbone_name,
            patch_size=patch_size,
            pose = pose
        )
        
        box_pair_predictor = nn.Linear(representation_size * 2, num_classes)
        box_pair_suppressor = nn.Linear(representation_size * 2, 1)

        interaction_head = InteractionHead(
            object_n_verb_to_interaction=object_n_verb_to_interaction,
            box_roi_pool=box_roi_pool,
            box_pair_head=box_pair_head,
            box_pair_suppressor=box_pair_suppressor,
            box_pair_predictor=box_pair_predictor,
            backbone=backbone,
            num_classes=num_classes,
            logit_scale=logit_scale,
            human_idx=human_idx,
            box_nms_thresh=box_nms_thresh,
            box_score_thresh=box_score_thresh,
            max_human=max_human,
            max_object=max_object,
            distributed=distributed,
            backbone_name=backbone_name,
            pose=pose
        )
    
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        if backbone_name == "resnet50":
            transform = HOINetworkTransform(min_size, max_size,
            image_mean, image_std)
        elif backbone_name == "CLIP" or backbone_name == "CLIP_CLS":
            transform  = Compose([
                        ToTensor()
                    ])

        super().__init__(backbone, backbone_name, interaction_head, transform, postprocess, rank, patch_size, human_idx, pose)