"""
Train and validate with distributed data parallel

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import random

from models import VIPLO
from utils import custom_collate, CustomisedDLE, DataFactory

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    trainset = DataFactory(
        name=args.dataset, partition=args.partitions[0],
        data_root=args.data_root,
        detection_root=args.train_detection_dir,
        flip=True, color_jitter=False, backbone_name=args.backbone_name, num_classes=args.num_class, pose=not args.poseoff
    )

    valset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root,
        detection_root=args.val_detection_dir, backbone_name=args.backbone_name, num_classes=args.num_class, pose=not args.poseoff
    )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank, seed=args.random_seed)
    )

    val_loader = DataLoader(
        dataset=valset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            valset, 
            num_replicas=args.world_size, 
            rank=rank)
    )

    # Fix random seed fo r model synchronisation
    torch.manual_seed(args.random_seed + rank)
    np.random.seed(args.random_seed + rank)
    random.seed(args.random_seed + rank)
    torch.backends.cudnn.benchmark = True
    
    

    object_to_target = train_loader.dataset.dataset.object_to_verb
    object_to_interaction = train_loader.dataset.dataset.object_to_interaction
    object_n_verb_to_interaction = train_loader.dataset.dataset.object_n_verb_to_interaction
    verb_list = train_loader.dataset.dataset.verbs
    human_idx = 49
    num_classes = args.num_class

    
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    net = VIPLO(
        object_to_target, object_n_verb_to_interaction, object_to_interaction, verb_list, human_idx, num_classes=num_classes, backbone_name=args.backbone_name, 
        output_size=args.roi_size,
        num_iterations=args.num_iter, postprocess=False,
        max_human=args.max_human, max_object=args.max_object,
        box_score_thresh=args.box_score_thresh,
        distributed=True, rank=rank, patch_size=args.patch_size, pose=not args.poseoff,
    )



    if os.path.exists(args.checkpoint_path):
        print("=> Rank {}: continue from saved checkpoint".format(
            rank), args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optim_state_dict = checkpoint['optim_state_dict']
        sched_state_dict = checkpoint['scheduler_state_dict']
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
    else:
        print("=> Rank {}: start from a randomly initialised model".format(rank))
        optim_state_dict = None
        sched_state_dict = None
        epoch = 0; iteration = 0

    engine = CustomisedDLE(
        net,
        train_loader,
        val_loader,
        num_classes=num_classes,
        backbone_name=args.backbone_name,
        print_interval=args.print_interval,
        cache_dir=args.cache_dir
    )
    # Seperate backbone parameters from the rest
    param_group_1 = []
    param_group_2 = []
    for k, v in engine.fetch_state_key('net').named_parameters():
        if v.requires_grad:
            if k.startswith('module.backbone'):
                param_group_1.append(v)
            elif k.startswith('module.interaction_head'):
                param_group_2.append(v)
            else:
                raise KeyError(f"Unknown parameter name {k}")
    # Fine-tune backbone with lower learning rate
    optim = torch.optim.AdamW([
        {'params': param_group_1, 'lr': args.learning_rate * args.lr_decay},
        {'params': param_group_2}
        ], lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    lambda1 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
    lambda2 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=[lambda1, lambda2]
    )
    # Override optimizer and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    engine.update_state_key(epoch=epoch, iteration=iteration)

    engine(args.num_epochs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=4, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--backbone-name', default='CLIP_CLS', type=str)
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--roi-size', default=7, type=int)
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--train-detection-dir', default='hicodet/detections/train2015_vitpose', type=str)
    parser.add_argument('--val-detection-dir', default='hicodet/detections/test2015_vitpose', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--num-epochs', default=8, type=int)
    parser.add_argument('--random-seed', default=42, type=int)
    parser.add_argument('--learning-rate', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=8, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--lr-decay', default=0.1, type=float,
                        help="The multiplier by which the learning rate is reduced")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--milestones', nargs='+', default=[6,], type=int,
                        help="The epoch number when learning rate is reduced")
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--print-interval', default=300, type=int)
    parser.add_argument('--checkpoint-path', default='', type=str)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints/train')
    parser.add_argument('--num-class', default=117, type=int)
    parser.add_argument('--poseoff', action='store_true')


    args = parser.parse_args()
    print(args)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))