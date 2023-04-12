"""
Test a model and compute detection mAP

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader

import pocket

from hicodet.hicodet import HICODet
from models import VIPLO
from utils import DataFactory, custom_collate, test

def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    num_anno = torch.tensor(HICODet(None, anno_file=os.path.join(
        args.data_root, 'instances_train2015.json')).anno_interaction)
    rare = torch.nonzero(num_anno < 10).squeeze(1)
    non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
    num_classes = args.num_class
    dataloader = DataLoader(
        dataset=DataFactory(
            name='hicodet', partition=args.partition,
            data_root=args.data_root,
            detection_root=args.detection_dir, backbone_name=args.backbone_name, num_classes=args.num_class, pose=not args.poseoff
        ), collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True
    )
    object_to_target = dataloader.dataset.dataset.object_to_verb
    object_n_verb_to_interaction = dataloader.dataset.dataset.object_n_verb_to_interaction
    object_to_interaction = dataloader.dataset.dataset.object_to_interaction
    verb_list = dataloader.dataset.dataset.verbs
    net = VIPLO(
        object_to_target, object_n_verb_to_interaction, object_to_interaction, verb_list, 49, num_classes = num_classes, backbone_name=args.backbone_name,
        output_size=args.roi_size, num_iterations=args.num_iter, max_human=args.max_human, max_object=args.max_object,
        box_score_thresh=args.box_score_thresh, patch_size=args.patch_size, pose=not args.poseoff
    )

    epoch = 0
    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint["epoch"]
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
            "Proceed to use a randomly initialised model.\n")

    net.cuda()
    timer = pocket.utils.HandyTimer(maxlen=1)
    
    with timer:
        test_ap = test(net, dataloader)
    print("Model at epoch: {} | time elapsed: {:.2f}s\n"
        "Full: {:.4f}, rare: {:.4f}, non-rare: {:.4f}".format(
        epoch, timer[0], test_ap.mean(),
        test_ap[rare].mean(), test_ap[non_rare].mean()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--detection-dir', default='hicodet/detections/test2015_gt_vitpose',
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--backbone-name', default='CLIP_CLS', type=str)
    parser.add_argument('--num-class', default=117, type=int)
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--roi-size', default=7, type=int)
    parser.add_argument('--poseoff', action='store_true')
    
    args = parser.parse_args()
    print(args)

    main(args)
