import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops
import numpy as np


from torch.nn import Module
from torch import nn, Tensor
from pocket.ops import Flatten
from typing import Optional, List, Tuple
from collections import OrderedDict

from ops import compute_spatial_encodings, binary_focal_loss, make_pose_box, compute_spatial_encodings_with_pose_to_attention

class InteractionHead(Module):
    """Interaction head that constructs and classifies box pairs
    Parameters:
    -----------
    box_roi_pool: Module
        Module that performs RoI pooling or its variants
    box_pair_head: Module
        Module that constructs and computes box pair features
    box_pair_suppressor: Module
        Module that computes unary weights for each box pair
    box_pair_predictor: Module
        Module that classifies box pairs
    human_idx: int
        The index of human/person class in all objects
    num_classes: int
        Number of target classes
    box_nms_thresh: float, default: 0.5
        Threshold used for non-maximum suppression
    box_score_thresh: float, default: 0.2
        Threshold used to filter out low-quality boxes
    max_human: int, default: 15
        Number of human detections to keep in each image
    max_object: int, default: 15
        Number of object (excluding human) detections to keep in each image
    distributed: bool, default: False
        Whether the model is trained under distributed data parallel. If True,
        the number of positive logits will be averaged across all subprocesses
    """
    def __init__(self,
        # Network components
        object_n_verb_to_interaction,
        box_roi_pool: Module,
        box_pair_head: Module,
        box_pair_suppressor: Module,
        box_pair_predictor: Module,
        backbone: Module, 
        # Dataset properties
        human_idx: int,
        num_classes: int,
        logit_scale,
        # Hyperparameters
        box_nms_thresh: float = 0.5,
        box_score_thresh: float = 0.2,
        max_human: int = 15,
        max_object: int = 15,
        # Misc
        distributed: bool = False,
        backbone_name: str = "resnet50",
        pose: bool = True
    ):
        super().__init__()
        if backbone_name != "resnet50":
            self.backbone = backbone
        self.object_n_verb_to_interaction = object_n_verb_to_interaction
        self.box_roi_pool = box_roi_pool
        self.box_pair_head = box_pair_head
        self.box_pair_suppressor = box_pair_suppressor
        self.box_pair_predictor = box_pair_predictor
        self.backbone_name = backbone_name
        self.pose = pose
        self.num_classes = num_classes
        self.human_idx = human_idx
        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.max_human = max_human
        self.max_object = max_object
        self.distributed = distributed

    def preprocess(self,
        detections: List[dict],
        targets: List[dict],
        append_gt: Optional[bool] = None
    ) :

        results = []
        for b_idx, detection in enumerate(detections):
            boxes = detection['boxes']
            labels = detection['labels']
            scores = detection['scores']
            
            if self.pose:
                human_joints = detection['human_joints']
                human_joints_score = detection['human_joints_score']

            original_human_index = (labels == self.human_idx).nonzero(as_tuple=True)[0]
            
            # Append ground truth during training
            if append_gt is None:
                append_gt = self.training
            if append_gt:
                target = targets[b_idx]
                n = target["boxes_h"].shape[0]
                boxes = torch.cat([target["boxes_h"], target["boxes_o"], boxes])
                scores = torch.cat([torch.ones(2 * n, device=scores.device), scores])
                labels = torch.cat([
                    self.human_idx * torch.ones(n, device=labels.device).long(),
                    target["object"],
                    labels
                ]).long()
                
                original_human_index = (labels == self.human_idx).nonzero(as_tuple=True)[0]
                
                if self.pose:
                    human_joints = torch.cat([target['human_joints'], human_joints])
                    human_joints_score = torch.cat([target['human_joints_score'], human_joints_score])
                    assert human_joints.shape[0] == human_joints_score.shape[0]
                    
                
            # Remove low scoring examples
            active_idx = torch.nonzero(
                scores >= self.box_score_thresh
            ).squeeze(1)
            # Class-wise non-maximum suppression
            
            keep_idx = box_ops.batched_nms(
                boxes[active_idx],
                scores[active_idx],
                labels[active_idx],
                self.box_nms_thresh
            )
            active_idx = active_idx[keep_idx]
            # Sort detections by scores
            sorted_idx = torch.argsort(scores[active_idx], descending=True)
            active_idx = active_idx[sorted_idx]
            # Keep a fixed number of detections
            h_idx = torch.nonzero(labels[active_idx] == self.human_idx).squeeze(1)
            o_idx = torch.nonzero(labels[active_idx] != self.human_idx).squeeze(1)
            if len(h_idx) > self.max_human:
                h_idx = h_idx[:self.max_human]
            if len(o_idx) > self.max_object:
                o_idx = o_idx[:self.max_object]
            # Permute humans to the top
            keep_idx = torch.cat([h_idx, o_idx])
            active_idx = active_idx[keep_idx]
            
            active_original_human_idx = active_idx[labels[active_idx]==self.human_idx]
            active_human_idx = (original_human_index.unsqueeze(1) == active_original_human_idx).nonzero(as_tuple=True)[0]

            if self.pose:
                results.append(dict(
                    boxes=boxes[active_idx].view(-1, 4), 
                    labels=labels[active_idx].view(-1),
                    scores=scores[active_idx].view(-1),
                    human_joints=human_joints[active_human_idx],
                    human_joints_score=human_joints_score[active_human_idx]
                ))
            else:
                results.append(dict(
                    boxes=boxes[active_idx].view(-1, 4), 
                    labels=labels[active_idx].view(-1),
                    scores=scores[active_idx].view(-1)
                ))

        return results
    
    def compute_interaction_classification_loss(self, results: List[dict]):
        scores = []; labels = []
        for result in results:
            scores.append(result['scores'])
            labels.append(result['labels'])

        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            torch.cat(scores), labels, reduction='sum', gamma=0.2
        )
        ##will not happen when batch size is large enough 
        if n_p == 0:
            n_p = 1

        return loss / n_p
        
    def compute_interactiveness_loss(self, results: List[dict]):
        weights = []; labels = []
        for result in results:
            weights.append(result['weights'])
            labels.append(result['unary_labels'])

        weights = torch.cat(weights)
        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            weights, labels, reduction='sum', gamma=2.0
        )
        ##will not happen when batch size is large enough 
        if n_p == 0:
            n_p = 1
        return loss / n_p

    def postprocess(self,
        logits_p: Tensor,
        logits_s: Tensor,
        prior: List[Tensor],
        boxes_h: List[Tensor],
        boxes_o: List[Tensor],
        object_class: List[Tensor],
        labels: List[Tensor]
    ):
        """
        Parameters:
        -----------
        logits_p: Tensor
            (N, K) Classification logits on each action for all box pairs
        logits_s: Tensor
            (N, 1) Logits for unary weights
        logits_ph: Tensor
            (N, 900) Phrase embedding for PhraseHOI. 
        prior: List[Tensor]
            Prior scores organised by images. Each tensor has shape (2, M, K).
            M could be different for different images
        boxes_h: List[Tensor]
            Human bounding box coordinates organised by images (M, 4)
        boxes_o: List[Tensor]
            Object bounding box coordinates organised by images (M, 4)
        object_classes: List[Tensor]
            Object indices for each pair organised by images (M,)
        labels: List[Tensor]
            Binary labels on each action organised by images (M, K)
        Returns:
        --------
        results: List[dict]
            Results organised by images, with keys as below
            `boxes_h`: Tensor[M, 4]
            `boxes_o`: Tensor[M, 4]
            `index`: Tensor[L]
                Expanded indices of box pairs for each predicted action
            `prediction`: Tensor[L]
                Expanded indices of predicted actions
            `scores`: Tensor[L]
                Scores for each predicted action
            `object`: Tensor[M]
                Object indices for each pair
            `prior`: Tensor[2, L]
                Prior scores for expanded pairs
            `weights`: Tensor[M]
                Unary weights for each box pair
            'phrase': Tensor[M, 900]
            `labels`: Tensor[L], optional
                Binary labels on each action
            `unary_labels`: Tensor[M], optional
                Labels for the unary weights
        """
        num_boxes = [len(b) for b in boxes_h]
        weights = torch.sigmoid(logits_s).squeeze(1)
        scores = torch.sigmoid(logits_p)
        weights = weights.split(num_boxes)
        scores = scores.split(num_boxes)
                    
        if len(labels) == 0:
            labels = [None for _ in range(len(num_boxes))]

        results = []
            
        for w, s, p, b_h, b_o, o, l in zip(
            weights, scores, prior, boxes_h, boxes_o, object_class, labels
        ):
            # Keep valid classes
            x, y = torch.nonzero(p[0]).unbind(1)

            result_dict = dict(
                boxes_h=b_h, boxes_o=b_o,
                index=x, prediction=y,
                scores=s[x, y] * p[:, x, y].prod(dim=0) * w[x].detach(),
                object=o, prior=p[:, x, y], weights=w
            )
            # If binary labels are provided
            if l is not None:
                result_dict['labels'] = l[x, y]
                result_dict['unary_labels'] = l.sum(dim=1).clamp(max=1)

            results.append(result_dict)

        return results
        
    def forward(self,
        features: OrderedDict,
        detections: List[dict],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[dict]] = None,
        images=None
    ):
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        detections: List[dict]
            Object detections with the following keys
            `boxes`: Tensor[N, 4]
            `labels`: Tensor[N]
            `scores`: Tensor[N]
        image_shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        targets: List[dict], optional
            Interaction targets with the following keys
            `boxes_h`: Tensor[G, 4]
            `boxes_o`: Tensor[G, 4]
            `object`: Tensor[G]
                Object class indices for each pair
            `labels`: Tensor[G]
                Target class indices for each pair
        Returns:
        --------
        results: List[dict]
            Results organised by images. During training the loss dict is appended to the
            end of the list, resulting in the length being larger than the number of images
            by one. For the result dict of each image, refer to `postprocess` for documentation.
            The loss dict has two keys
            `hoi_loss`: Tensor
                Loss for HOI classification
            `interactiveness_loss`: Tensor
                Loss incurred on learned unary weights
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
            
        detections = self.preprocess(detections, targets)

        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        box_scores = [detection['scores'] for detection in detections]
        box_segs = None
        if self.pose:
            human_joints = [detection['human_joints'] for detection in detections]
            human_joints_score = [detection['human_joints_score'] for detection in detections]
        else:
            human_joints = [None for _ in detections]

        if self.backbone_name == "resnet50" or self.backbone_name=="CLIP" or self.backbone_name == "GLIP":
            box_features = self.box_roi_pool(features, box_coords, image_shapes)
            pose_box_features = None
 
        elif self.backbone_name == "CLIP_CLS":
            if self.pose:
                box_features, global_features, patch_features = self.backbone.encode_image(images, box_coords, box_segs, need_patch=True)
                features['global'] = global_features
                B, L, C = patch_features.shape
                features['0'] = patch_features.permute(0,2,1).reshape(B, C, int(L**0.5), int(L**0.5)).to(dtype=torch.float32)
                image_hw = images.shape[2:]
                pose_box_coords = make_pose_box(box_coords, box_labels, human_joints, self.human_idx, image_hw)
                pose_box_features = self.box_roi_pool(features, pose_box_coords, image_shapes)
            else:
                box_features, global_features = self.backbone.encode_image(images, box_coords, box_segs, need_patch=False)
                features['global'] = global_features
                pose_box_features = None
        

        box_pair_features, box_pair_local_features, boxes_h, boxes_o, object_class,\
        box_pair_labels, box_pair_prior = self.box_pair_head(
            features, image_shapes, box_features, pose_box_features,
            box_coords, box_labels, box_scores, human_joints, human_joints_score, targets
        )

        box_pair_features = torch.cat(box_pair_features)
        if self.pose:
            logits_p = self.box_pair_predictor(box_pair_features)
        else:
            logits_p = self.box_pair_predictor(box_pair_features)

        logits_s = self.box_pair_suppressor(box_pair_features)
            
        results = self.postprocess(
            logits_p, logits_s, box_pair_prior,
            boxes_h, boxes_o,
            object_class, box_pair_labels
        )

        if self.training:
            loss_dict = dict(
            hoi_loss=self.compute_interaction_classification_loss(results),
            interactiveness_loss=self.compute_interactiveness_loss(results)
        )
            results.append(loss_dict)

        return results

class MultiBranchFusion(Module):
    """
    Multi-branch fusion module
    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        representation_size: int, cardinality: int
    ):
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(representation_size / cardinality)
        assert sub_repr_size * cardinality == representation_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, representation_size)
            for _ in range(cardinality)
        ])
    def forward(self, appearance: Tensor, spatial: Tensor):
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))

class MessageMBF(MultiBranchFusion):
    """
    MBF for the computation of anisotropic messages
    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    node_type: str
        Nature of the sending node. Choose between `human` amd `object`
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int,
        spatial_size: int,
        representation_size: int,
        node_type: str,
        cardinality: int
    ):
        super().__init__(appearance_size, spatial_size, representation_size, cardinality)

        if node_type == 'human':
            self._forward_method = self._forward_human_nodes
        elif node_type == 'object':
            self._forward_method = self._forward_object_nodes
        elif node_type == 'object_local_pose':
            self._forward_method = self._forward_object_local_pose_nodes
        else:
            raise ValueError("Unknown node type \"{}\"".format(node_type))

    def _forward_human_nodes(self, appearance: Tensor, spatial: Tensor):
        n_h, n = spatial.shape[:2]
        assert len(appearance) == n_h, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n, 1, 1)
                * fc_2(spatial).permute([1, 0, 2])
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)
        
    def _forward_object_nodes(self, appearance: Tensor, spatial: Tensor):
        n_h, n = spatial.shape[:2]
        assert len(appearance) == n, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n_h, 1, 1)
                * fc_2(spatial)
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)
        
    def _forward_object_local_pose_nodes(self, local_feature: Tensor, appearance: Tensor, spatial: Tensor):
        n_h, n = spatial.shape[:2]
        assert local_feature.shape[0] == n_h
        assert local_feature.shape[1] == n
        assert len(appearance) == n, "Incorrect size of dim0 for appearance features"
        #print("doing local pose")
        concat_appearance = torch.cat([local_feature, appearance.repeat(n_h, 1, 1)], dim=2)
        return torch.stack([
            fc_3(F.relu(
                fc_1(concat_appearance)
                * fc_2(spatial)
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)


    def forward(self, *args):
        return self._forward_method(*args)

class GraphHead(Module):
    """
    Graphical model head
    Parameters:
    -----------
    output_channels: int
        Number of output channels of the backbone
    roi_pool_size: int
        Spatial resolution of the pooled output
    node_encoding_size: int
        Size of the node embeddings
    num_cls: int
        Number of targe classes
    human_idx: int
        The index of human/person class in all objects
    object_class_to_target_class: List[list]
        The mapping (potentially one-to-many) from objects to target classes
    fg_iou_thresh: float, default: 0.5
        The IoU threshold to identify a positive example
    num_iter: int, default 2
        Number of iterations of the message passing process
    """
    def __init__(self,
        verb_list,
        backbone: Module,
        out_channels: int,
        roi_pool_size: int,
        node_encoding_size: int, 
        representation_size: int, 
        num_cls: int, human_idx: int,
        object_class_to_target_class: List[list],
        object_class_to_interaction_class: List[list],
        fg_iou_thresh: float = 0.5,
        num_iter: int = 2,
        backbone_name: str = 'resnet50',
        patch_size: int = 32, 
        pose: bool = True
    ):

        super().__init__()
        self.verb_list = verb_list
        if backbone_name != 'resnet50':
            self.backbone = backbone
        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size
        self.backbone_name = backbone_name
        self.num_cls = num_cls
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class
        self.object_class_to_interaction_class = object_class_to_interaction_class
        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter
        self.pose = pose
        
        # Box head to map RoI features to low dimensional
        if self.backbone_name == "CLIP_CLS":
            self.box_head = nn.Sequential(nn.Linear(out_channels, node_encoding_size),
                                        nn.ReLU(),
                                        nn.Linear(node_encoding_size, node_encoding_size),
                                        nn.ReLU())
            
            if self.pose:
                self.pose_head = nn.Sequential(
                    Flatten(start_dim=1),
                    nn.Linear(out_channels * 5 ** 2, node_encoding_size),
                    nn.ReLU(),
                    nn.Linear(node_encoding_size, node_encoding_size),
                    nn.ReLU()
                )

        else:
            self.box_head = nn.Sequential(
                Flatten(start_dim=1),
                nn.Linear(out_channels * roi_pool_size ** 2, node_encoding_size),
                nn.ReLU(),
                nn.Linear(node_encoding_size, node_encoding_size),
                nn.ReLU()
            )

        # Compute adjacency matrix
        self.adjacency = nn.Linear(representation_size, 1)

        # Compute messages
        self.sub_to_obj = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='human',
            cardinality=16
        )
        if self.pose:
            self.obj_to_sub = MessageMBF(
                node_encoding_size*2, 1024,
                representation_size, node_type= 'object_local_pose',
                cardinality=16
            )
        else:
            self.obj_to_sub = MessageMBF(
                node_encoding_size, 1024,
                representation_size, node_type= 'object', 
                cardinality=16
            )
            
        self.norm_h = nn.LayerNorm(node_encoding_size)
        self.norm_o = nn.LayerNorm(node_encoding_size)

        if self.pose:
            self.spatial_head = nn.Sequential(
                    nn.Linear(36, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1024),
                    nn.ReLU(),
                )
            self.joint_head = nn.Sequential(
                    nn.Linear(12, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1024),
                    nn.ReLU(),
                )
        else:
            self.spatial_head = nn.Sequential(
                nn.Linear(36, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 1024),
                nn.ReLU(),
            )

        self.attention_head = MultiBranchFusion(
            node_encoding_size * 2,
            1024, representation_size,
            cardinality=16
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Attention head for global features
        if self.backbone_name == 'resnet50' or self.backbone_name == "GLIP":
            self.attention_head_g = MultiBranchFusion(
                256, 1024,
                representation_size, cardinality=16
            )
        elif self.backbone_name == 'CLIP' or self.backbone_name  == "CLIP_CLS":
            self.attention_head_g = MultiBranchFusion(
                out_channels, 1024,
                representation_size, cardinality=16
            )

    def associate_with_ground_truth(self,
        boxes_h: Tensor,
        boxes_o: Tensor,
        targets: List[dict]
    ):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_cls, device=boxes_h.device)

        if len(targets['boxes_h'].shape)==2:
            x, y = torch.nonzero(torch.min(
                box_ops.box_iou(boxes_h, targets["boxes_h"]),
                box_ops.box_iou(boxes_o, targets["boxes_o"])
            ) >= self.fg_iou_thresh).unbind(1)

            labels[x, targets["labels"][y]] = 1

        return labels

    def compute_prior_scores(self,
        x: Tensor, y: Tensor,
        scores: Tensor,
        object_class: Tensor
    ):
        """
        Parameters:
        -----------
            x: Tensor[M]
                Indices of human boxes (paired)
            y: Tensor[M]
                Indices of object boxes (paired)
            scores: Tensor[N]
                Object detection scores (before pairing)
            object_class: Tensor[N]
                Object class indices (before pairing)
        """
        prior_h = torch.zeros(len(x), self.num_cls, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])
    
    def forward(self,
        features: OrderedDict, image_shapes: List[Tuple[int, int]],
        box_features: Tensor, pose_box_features: Tensor, box_coords: List[Tensor], 
        box_labels: List[Tensor], box_scores: List[Tensor], human_joints: List[Tensor], human_joints_score: List[Tensor],
        targets: Optional[List[dict]] = None, backbone = None, images = None, pose_box_coords = None, 
    ):
        """
        Parameters:
        -----------
            features: OrderedDict
                Feature maps returned by FPN
            box_features: Tensor
                (N, C, P, P) Pooled box features
            image_shapes: List[Tuple[int, int]]
                Image shapes, heights followed by widths
            box_coords: List[Tensor]
                Bounding box coordinates organised by images
            box_labels: List[Tensor]
                Bounding box object types organised by images
            box_scores: List[Tensor]
                Bounding box scores organised by images
            targets: List[dict]
                Interaction targets with the following keys
                `boxes_h`: Tensor[G, 4]
                `boxes_o`: Tensor[G, 4]
                `labels`: Tensor[G]
        Returns:
        --------
            all_box_pair_features: List[Tensor]
            all_boxes_h: List[Tensor]
            all_boxes_o: List[Tensor]
            all_object_class: List[Tensor]
            all_labels: List[Tensor]
            all_prior: List[Tensor]
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        if self.backbone_name == 'resnet50':
            global_features = self.avg_pool(features['3']).flatten(start_dim=1)
        elif self.backbone_name == "GLIP":
            global_features = self.avg_pool(features["4"]).flatten(start_dim=1)
        elif self.backbone_name == "CLIP":
            global_features = features['3']
        elif self.backbone_name == "CLIP_CLS":
            global_features = features['global']
        else:
            assert False, "Not supported backbone!"
        
        box_features = self.box_head(box_features) 
        if self.pose:
            pose_box_features = self.pose_head(pose_box_features)
            pose_box_features = pose_box_features.reshape(-1, 17, pose_box_features.shape[-1])
        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]
        
        counter = 0
        counter_h = 0
        all_boxes_h = []; all_boxes_o = []; all_object_class = []
        all_labels = []; all_prior = []
        all_box_pair_features = []
        all_box_pair_local_features = []
        for b_idx, (coords, labels, scores, human_joint, human_joint_score) in enumerate(zip(box_coords, box_labels, box_scores, human_joints, human_joints_score)):
            n = num_boxes[b_idx]
            device = box_features.device
            
            n_h = torch.sum(labels == self.human_idx).item()
            if self.pose:
                assert human_joint.shape[0] == n_h
            # Skip image when there are no detected human or object instances
            # and when there is only one detected instance
            if n_h == 0 or n <= 1:
                if self.pose:
                    all_box_pair_local_features.append(torch.zeros(
                        0, 1 * self.representation_size,
                        device=device)
                    )
                all_box_pair_features.append(torch.zeros(
                    0, 2 * self.representation_size,
                    device=device)
                )
                all_boxes_h.append(torch.zeros(0, 4, device=device))
                all_boxes_o.append(torch.zeros(0, 4, device=device))
                all_object_class.append(torch.zeros(0, device=device, dtype=torch.int64))
                all_prior.append(torch.zeros(2, 0, self.num_cls, device=device))
                all_labels.append(torch.zeros(0, self.num_cls, device=device))
                continue
            if not torch.all(labels[:n_h]==self.human_idx):
                raise ValueError("Human detections are not permuted to the top")

            # visual feature 의미 (human + object )
            node_encodings = box_features[counter: counter+n]
            # Duplicate human nodes
            h_node_encodings = node_encodings[:n_h]
            if self.pose:
                pose_box_feature = pose_box_features[counter_h: counter_h+n_h]
            
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(
                torch.arange(n_h, device=device),
                torch.arange(n, device=device)
            )
            # Remove pairs consisting of the same human instance
            x_keep, y_keep = torch.nonzero(x != y).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            # Human nodes have been duplicated and will be treated independently
            # of the humans included amongst object nodes
            x = x.flatten(); y = y.flatten()
            # Compute spatial features
            if self.pose:
                spatial_query, pose_key = compute_spatial_encodings_with_pose_to_attention([coords[x]], [coords[y]], [human_joint.repeat_interleave(n, dim=0)],[image_shapes[b_idx]])
                
            else:
                box_pair_spatial_semantic = compute_spatial_encodings(
                [coords[x]], [coords[y]], [image_shapes[b_idx]]
            )

            box_pair_spatial_semantic = self.spatial_head(spatial_query)
            pose_key_mat = self.joint_head(pose_key)

            
            if self.pose:
                pose_attention = torch.matmul(box_pair_spatial_semantic.unsqueeze(1), pose_key_mat.permute(0,2,1)).squeeze(1) * human_joint_score.repeat_interleave(n, dim=0)
                pose_attention_weight = pose_attention.softmax(dim=-1)

                pose_box_feature = pose_box_feature.repeat_interleave(n, dim=0)
                pose_local_feature = torch.sum(pose_box_feature * pose_attention_weight.unsqueeze(-1), dim=1)
                    
                pose_local_feature_reshaped = pose_local_feature.reshape(n_h, n, -1)


            # Reshape the spatial features
            box_pair_spatial_semantic_reshaped = box_pair_spatial_semantic.reshape(n_h, n, -1)
            adjacency_matrix = torch.ones(n_h, n, device=device)
            for _ in range(self.num_iter):
                # Compute weights of each edge
                weights = self.attention_head(
                    torch.cat([
                        h_node_encodings[x],
                        node_encodings[y]
                    ], 1),
                    box_pair_spatial_semantic
                )
                adjacency_matrix = self.adjacency(weights).reshape(n_h, n)

                # update local human nodes

                # Update human nodes
                if self.pose:
                    messages_to_h = F.relu(torch.sum(
                        adjacency_matrix.softmax(dim=1)[..., None] *
                        self.obj_to_sub(
                            pose_local_feature_reshaped,
                            node_encodings,
                            box_pair_spatial_semantic_reshaped
                        ), dim=1)
                    )
                else:
                    messages_to_h = F.relu(torch.sum(
                        adjacency_matrix.softmax(dim=1)[..., None] *
                        self.obj_to_sub(
                            node_encodings,
                            box_pair_spatial_semantic_reshaped
                        ), dim=1)
                    )

                h_node_encodings = self.norm_h(
                    h_node_encodings + messages_to_h
                )
                
                #Update object nodes (including human nodes)
                messages_to_o = F.relu(torch.sum(
                    adjacency_matrix.t().softmax(dim=1)[..., None] *
                    self.sub_to_obj(
                        h_node_encodings,
                        box_pair_spatial_semantic_reshaped
                    ), dim=1)
                )
                node_encodings = self.norm_o(
                    node_encodings + messages_to_o
                )

            if targets is not None:
                all_labels.append(self.associate_with_ground_truth(
                    coords[x_keep], coords[y_keep], targets[b_idx])
                )

            all_box_pair_features.append(torch.cat([
                    self.attention_head(
                        torch.cat([
                            h_node_encodings[x_keep],
                            node_encodings[y_keep]
                            ], 1),
                        box_pair_spatial_semantic_reshaped[x_keep, y_keep]
                    ), self.attention_head_g(
                        global_features[b_idx, None],
                        box_pair_spatial_semantic_reshaped[x_keep, y_keep])
                ], dim=1))

            all_boxes_h.append(coords[x_keep])
            all_boxes_o.append(coords[y_keep])
            all_object_class.append(labels[y_keep])

            all_prior.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            counter += n
            counter_h += n_h

        return all_box_pair_features, all_box_pair_local_features, all_boxes_h, all_boxes_o, \
           all_object_class, all_labels, all_prior