"""
Opearations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops
import cv2 
import numpy as np
import time

from torch import Tensor
from typing import List, Tuple

def compute_spatial_encodings(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
):
    """
    Parameters:
    -----------
        boxes_1: List[Tensor]
            First set of bounding boxes (M, 4)
        boxes_1: List[Tensor]
            Second set of bounding boxes (M, 4)
        shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        eps: float
            A small constant used for numerical stability

    Returns:
    --------
        Tensor
            Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)

def compute_spatial_encodings_with_pose(
    boxes_1: List[Tensor], boxes_2: List[Tensor], human_joints: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
):
    """
    Parameters:
    -----------
        boxes_1: List[Tensor]
            First set of bounding boxes (M, 4)
        boxes_1: List[Tensor]
            Second set of bounding boxes (M, 4)
        shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        eps: float
            A small constant used for numerical stability

    Returns:
    --------
        Tensor
            Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape, human_joint in zip(boxes_1, boxes_2, shapes, human_joints):
        h, w = shape
        if len(human_joint)!=0:
            human_joint[..., 0] = human_joint[..., 0].clip(0, w)
            human_joint[..., 1] = human_joint[..., 1].clip(0, h)
        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        human_joint_x = human_joint[:, :, 0]
        human_joint_y = human_joint[:, :, 1]
        human_joint_d_x = torch.abs(c2_x.unsqueeze(-1) - human_joint_x) / (b1_w.unsqueeze(-1) + eps)
        human_joint_d_y = torch.abs(c2_y.unsqueeze(-1) - human_joint_y) / (b1_h.unsqueeze(-1) + eps)

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)
        g = torch.cat([human_joint_x / w, human_joint_y / h, (c2_x.unsqueeze(-1) > human_joint_x).float() * human_joint_d_x, (c2_x.unsqueeze(-1) < human_joint_x).float() * human_joint_d_x,
        (c2_y.unsqueeze(-1) > human_joint_y).float() * human_joint_d_y, (c2_y.unsqueeze(-1) < human_joint_y).float() * human_joint_d_y], 1)

        h = torch.cat([f, g], dim=1)
        features.append(
            torch.cat([h, torch.log(h + eps)], 1)
        )
    return torch.cat(features)


def compute_spatial_encodings_with_pose_to_attention(
    boxes_1: List[Tensor], boxes_2: List[Tensor], human_joints: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
):
    """
    Parameters:
    -----------
        boxes_1: List[Tensor]
            First set of bounding boxes (M, 4)
        boxes_1: List[Tensor]
            Second set of bounding boxes (M, 4)
        shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        eps: float
            A small constant used for numerical stability

    Returns:
    --------
        Tensor
            Computed spatial encodings between the boxes (N, 36)
    """
    spatial_query = []
    pose_key = []
    for b1, b2, shape, human_joint in zip(boxes_1, boxes_2, shapes, human_joints):
        h, w = shape
        if len(human_joint)!=0:
            human_joint[..., 0] = human_joint[..., 0].clip(0, w)
            human_joint[..., 1] = human_joint[..., 1].clip(0, h)
        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        human_joint_x = human_joint[:, :, 0]
        human_joint_y = human_joint[:, :, 1]
        human_joint_d_x = torch.abs(c2_x.unsqueeze(-1) - human_joint_x) / (b1_w.unsqueeze(-1) + eps)
        human_joint_d_y = torch.abs(c2_y.unsqueeze(-1) - human_joint_y) / (b1_h.unsqueeze(-1) + eps)

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)
        g = torch.cat([human_joint_x / w, human_joint_y / h, (c2_x.unsqueeze(-1) > human_joint_x).float() * human_joint_d_x, (c2_x.unsqueeze(-1) < human_joint_x).float() * human_joint_d_x,
        (c2_y.unsqueeze(-1) > human_joint_y).float() * human_joint_d_y, (c2_y.unsqueeze(-1) < human_joint_y).float() * human_joint_d_y], 1)

        g = g.reshape(-1, 6, 17).permute(0, 2, 1)

        pose_key.append(torch.cat([g, torch.log(g+eps)], 2))

        spatial_query.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(spatial_query), torch.cat(pose_key)





def binary_focal_loss(
    x: Tensor, y: Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: float = 1e-6
):
    loss = (1 - y - alpha).abs() * ((y-x).abs() + eps) ** gamma * \
        torch.nn.functional.binary_cross_entropy(
            x, y, reduction='none'
        )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction method {}".format(reduction))


def generate_pose_heatmap(human_bbox: Tensor, human_joints : Tensor, human_joints_score: Tensor):
    if len(human_bbox) == 0:
        return torch.tensor([]).to(human_bbox.device)
    EPS = 1e-3
    SIGMA = 0.6
    PADDING = 3
    heatmap_wh = torch.tensor([28, 56]).to(human_bbox.device)

    num_box = human_bbox.shape[0]
    assert num_box == human_joints.shape[0]
    assert num_box == human_joints_score.shape[0]

    human_joints_score = torch.clip(human_joints_score, min=EPS)

    min_joints = torch.min(human_joints, dim=1).values
    max_joints = torch.max(human_joints, dim=1).values
    pose_wh = max_joints - min_joints

    scale_wh = (heatmap_wh - PADDING * 2) / pose_wh

    resized_human_joints = (human_joints - min_joints.unsqueeze(1)) * scale_wh.unsqueeze(1) + PADDING
    x = torch.arange(0, heatmap_wh[0]).to(torch.float32).to(human_bbox.device)
    y = torch.arange(0, heatmap_wh[1]).to(torch.float32).to(human_bbox.device).unsqueeze(-1)
    heatmap = torch.exp(-((x - resized_human_joints[..., 0].unsqueeze(-1).unsqueeze(-1))**2 + (y - resized_human_joints[..., 1].unsqueeze(-1).unsqueeze(-1))**2) / 2 / SIGMA**2)
    heatmap = torch.where(heatmap>EPS, heatmap, torch.tensor(0, dtype=heatmap.dtype, device=heatmap.device))

    heatmap = heatmap * human_joints_score.unsqueeze(-1).unsqueeze(-1)

    return heatmap


#### human_joint clipping
def make_pose_box(box_coords, box_labels, human_joints, human_idx, image_hw, gamma=0.3):
    height = image_hw[0]
    width = image_hw[1]
    result = []
    batch_size = len(box_coords)
    for b_i in range(batch_size):
        box_coord = box_coords[b_i]
        box_label = box_labels[b_i]
        human_joint = human_joints[b_i]
        human_box_coord = box_coord[box_label == human_idx]
        if len(human_joint)!=0:
            human_joint[..., 0] = human_joint[..., 0].clip(0, width)
            human_joint[..., 1] = human_joint[..., 1].clip(0, height)
        num_human_box = len(human_box_coord)
        human_box_height = human_box_coord[:, 3] - human_box_coord[:, 1]
        pose_box_size = (human_box_height * gamma).unsqueeze(-1).unsqueeze(-1)
        human_pose_box_coord = torch.cat([human_joint - pose_box_size / 2, human_joint + pose_box_size / 2], dim=-1).reshape(num_human_box*17, 4)

        result.append(human_pose_box_coord)
    
    return result


def warpaffine_image(image, n_px, device):
    
    width, height = image.size
    #height, width = image.shape[1:]
    #image_size = np.array([n_px, n_px])
    image_size = torch.tensor([n_px, n_px]).to(device)
    x,y = torch.tensor(0).to(device), torch.tensor(0).to(device)
    #x,y = 0, 0
    #w = width
    #h = height 
    w = torch.tensor(width-1).to(device)
    h = torch.tensor(height-1).to(device)
    #x1 = max(0, x)
    #y1 = max(0, y)
    #x2 = min(width - 1, x1 + max(0, w - 1))
    #y2 = min(height - 1, y1 + max(0, h - 1))
    #x1 = torch.minimum(width-1, x+ )
    #x,y,w,h = x1, y1, x2 - x1, y2 - y1

    aspect_ratio = image_size[0] / image_size[1]
    #center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    center = torch.tensor([x + w * 0.5, y + h * 0.5], dtype=torch.float32).to(device)
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    #scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    scale = torch.tensor([w / 200.0, h / 200.0], dtype=torch.float32).to(device)
    trans = get_warp_matrix(center * 2.0, image_size - 1.0, scale * 200.0, device)
    processed_img = cv2.warpAffine(
                        np.array(image),
                        np.array(trans.cpu()), (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR)
    img_meta = {'center': center, 'scale': scale, 'n_px': n_px}


    return processed_img, trans, img_meta

def get_warp_matrix(size_input, size_dst, size_target, device):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    # theta = np.deg2rad(theta) theta:0
    #matrix = np.zeros((2, 3), dtype=np.float32)
    matrix = torch.zeros((2,3), dtype=torch.float32).to(device)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    #matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 0] = 1.0 * scale_x
    #matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 1] = -0.0 * scale_x 
    # matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
    #                           0.5 * size_input[1] * math.sin(theta) +
    #                           0.5 * size_target[0])
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * 1.0 +
                               0.5 * size_input[1] * 0.0 +
                              0.5 * size_target[0])
    # matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 0] = 0.0 * scale_y
    # matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 1] = 1.0 * scale_y
    # matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
    #                           0.5 * size_input[1] * math.cos(theta) +
    #                           0.5 * size_target[1])
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * 0.0 -
                               0.5 * size_input[1] * 1.0 +
                               0.5 * size_target[1])
    return matrix


def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        np.ndarray[..., 2]: Result coordinate of joints.
    """
    #joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return torch.matmul(torch.cat((joints, joints[:, 0:1] * 0 +1), dim=1), mat.T).reshape(shape)
    #return np.dot(
    #    np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
    #    mat.T).reshape(shape)
def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = torch.ones_like(coords).to(coords.device)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords