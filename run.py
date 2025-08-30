# Jiayu Xiong re-impt @ 2025.7.15
# fork from https://github.com/YuanGongND/ast

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from timm.data import Mixup
from torch.utils.data import WeightedRandomSampler

import models.rwkv7
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate
from timm.data.random_erasing import RandomErasing
from timm.models.vision_transformer import VisionTransformer
print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--n_print_steps', help='print frequency', default=2000, type=bool)
parser.add_argument("--exp-dir", type=str, default="expr_t_scan", help="directory to dump experiments")
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument('--save_model', help='save the model or not', default=True, type=bool)

parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("--blr", type=float, default=2e-5, metavar="LR", help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
parser.add_argument("--min_lr", type=float, default=1e-7, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0")
parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)")
parser.add_argument("--epochs", type=int, default=25, help="Total number of epochs to train")
parser.add_argument("--accum_iter", type=int, default=32, help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")

# Architecture
parser.add_argument("--n_layer", type=int, default=12, help="Number of layers (default: 12)")
parser.add_argument("--n_embd", type=int, default=192, help="Embedding dimension size (default: 768)")
parser.add_argument("--ctx_len", type=int, default=2048, help="Context length / sequence length (default: 4096)")
parser.add_argument("--head_size_a", type=int, default=64, help="Internal head size A (do not change; default: 64)")
parser.add_argument("--head_size_divisor", type=int, default=8, help="Internal head size divisor (do not change; default: 8)")

# Dataset
parser.add_argument("--data-train", type=str, default='un_train_index_cleaned.csv', help="training data json")
parser.add_argument("--data-val", type=str, default='eval_index_cleaned.csv', help="validation data json")
parser.add_argument("--root-train", type=str, default='unbal/', help="training data root path")
parser.add_argument("--root-val", type=str, default='eval/', help="validation data root path")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='class_labels_indices.csv', help="csv with class labels")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")
parser.add_argument("--num_classes", type=int, default=527, help="Number of output classes")
parser.add_argument("--dataset_mean", type=float, default=-4.421761, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.32408, help="the dataset spectrogram std")
parser.add_argument("--target_length", type=int, default=992, help="the dataset time-axis length")
parser.add_argument("--h", type=int, default=62, help="auxiliary parameters help convolution unfold, h")
parser.add_argument("--w", type=int, default=8, help="Auxiliary parameters help convolution unfold, w")
parser.add_argument("--num_mels", type=int, default=128, help="the dataset freq-axis length")
parser.add_argument("--bal", type=str, default=True, help="use balanced sampling or not")

parser.add_argument("--metrics", type=str, default="mAP", help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default="BCE", help="loss function", choices=["BCE", "CE"])

parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")

# Mixup params
parser.add_argument('--smoothing', type=float, default=0.1, help='label smoothing')
parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0.2, help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--re_prob', type=float, default=0.25, help='random erasing prob.')
parser.add_argument('--drop', type=float, default=0.15, help='drop path prob.')

args = parser.parse_args()
def main():
    mixup_fn, re_fn = None, None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
    if args.re_prob > 0:
        re_fn = RandomErasing(probability=args.re_prob, max_area=0.2)

    audio_conf = {'target_length': args.target_length, 'melbins':args.num_mels, 'skip_norm': False, 'mean': args.dataset_mean, 'std': args.dataset_std, 'mode': 'train', 'dataset': 'audioset', 'root': args.root_train}
    val_audio_conf = {'target_length': args.target_length, 'melbins':args.num_mels, 'skip_norm': False, 'mean': args.dataset_mean, 'std': args.dataset_std, 'mode': 'test', 'dataset': 'audioset', 'root': args.root_val}
    if args.bal == True:
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-4]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetSpec(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True) #,
            # prefetch_factor=8*int(args.batch_size//args.num_workers), persistent_workers=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetSpec(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True) #,
            # prefetch_factor=8*int(args.batch_size//args.num_workers), persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetSpec(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) #,
        # prefetch_factor=8*int(args.batch_size//args.num_workers), persistent_workers=True)
    audio_model = models.rwkv7.RWKV(args).bfloat16()# .half()

    print("\nCreating experiment directory: %s" % args.exp_dir)
    # os.makedirs("%s/models" % args.exp_dir)
    # you can uncomment, but when the folder exists, it can no longer point to the same directory for training
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

    print('Now starting training for {:d} epochs'.format(args.epochs))
    train(audio_model, train_loader, val_loader, mixup_fn, re_fn, args)

    return



main()