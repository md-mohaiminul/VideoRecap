import os
import argparse
import time
import math
import sys
from tqdm import tqdm
import wandb

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms.functional import InterpolationMode

from collections import OrderedDict
from transformers import AutoTokenizer

from src.data.video_transforms import Permute
import src.utils.distributed as dist_utils
from src.configs.defaults import defaultConfigs
from src.models.video_recap import VideoRecap
from src.data.datasets import VideoCaptionDataset, CaptionDataCollator
from src.utils.scheduler import cosine_scheduler
from src.utils.random import random_seed

def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)
    
def main(args):
    print(args)
    dist_utils.init_distributed_mode(args)
    
    random_seed(args.seed, dist_utils.get_rank())
    
    # Training data
    print("=> Creating train dataset")
    if args.video_feature_type=='pixel':
        if args.vision_model_type=='clip_b16':
            crop_size = 224
            train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (crop_size, crop_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.Normalize(mean = (0.48145466, 0.4578275, 0.40821073), 
                                     std = (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        else:
            crop_size = 224
            transforms_list = [
                Permute([3, 0, 1, 2]),    # T H W C -> C T H W
                transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
                transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]),
            ]
            train_transform = transforms.Compose(transforms_list)
    else:
        train_transform = None
    
    train_dataset = VideoCaptionDataset(args, transform = train_transform, is_training=True)
    print('len(train_dataset) = {}'.format(len(train_dataset)))

    
    if args.metadata_pseudo is not None:
        metadata = args.metadata
        args.metadata = args.metadata_pseudo
        aux_dataset = VideoCaptionDataset(args, transform = train_transform)
        print('len(aux_dataset) = {}'.format(len(aux_dataset)))
        train_dataset = torch.utils.data.ConcatDataset([aux_dataset, train_dataset])
        args.metadata = metadata
        print('len(train_dataset)+len(aux_dataset) = {}'.format(len(train_dataset)))
    
    # for i in range(10):
    #     s = train_dataset.__getitem__(i)
    #     print(s.keys())
    #     print(s['index'], s['caption'])
    #     print(s['video_features'].shape, torch.sum(s['video_features']))
    
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    
    collator = CaptionDataCollator(tokenizer, max_gen_tokens = args.max_gen_tokens,
                                    add_bos = True, add_eos = True, pad_token_id = 0)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collator, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True
    )
    print('len(train_loader) = {}'.format(len(train_loader)))
    
    # for sample in train_loader:
    #     for k in sample:
    #         print(k, sample[k].shape)
    #     print(torch.sum(sample['video_features']))
    #     break
    
    print("=> Creating model")
    model = VideoRecap(args)

    if args.distributed:
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=False
        )
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total     parameters :', total_params)
    print('Trainable parameters :', trainable_params)
    
        
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
            

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    if args.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=torch.optim.AdamW,
            lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd
        )
    else:
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                      eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch 
            if args.distributed:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k.replace('module.', '')] = v

            found, missing = [], []
            for k, v in state_dict.items():
                if 'video_queries' in k and v.shape[0]!=args.num_video_queries:
                    step = v.shape[0] / (args.num_video_queries + 1)
                    idx = np.arange(0, v.shape[0], step).astype(int)[1:]
                    state_dict[k] = v[idx]
                elif 'text_queries' in k and v.shape[0]!=args.num_text_queries:
                    step = v.shape[0] / (args.num_text_queries + 1)
                    idx = np.arange(0, v.shape[0], step).astype(int)[1:]
                    state_dict[k] = v[idx]
                
                if k in model.state_dict():
                    found.append(k)
                else:
                    missing.append(k)
                    
            print('Pretrained weights found', len(found))
            print('Missing weights', len(missing))

            model.load_state_dict(state_dict, strict=False)
            
            # try:
            #     state_dict_optim = checkpoint['optimizer']
            #     for k, v in state_dict_optim.items():
            #         print(k, v)
            #     optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            # except:
            #     print('Cannot load optimizer!')
                
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    cudnn.benchmark = True
    
    if args.lr_scheduler_type == 'cosine':
        lr_schedule = cosine_scheduler(
            args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start,
        )
    elif args.lr_scheduler_type == 'linear':
            lr_decay = args.lr_decay
            lr_step_size = args.lr_step_size
            lr_schedule = StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay)
    else:
        lr_schedule = None
        
    if dist_utils.is_main_process() and args.wandb:
        wandb_run = wandb.init(project = 'VideoCapHier',
                        config = args,
                        save_code = True,
                        group = 'caption',
            )
        
    print("=> beginning training")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train(train_loader, model, optimizer, scaler, epoch, lr_schedule, args)
        
        if args.lr_scheduler_type == 'linear':
            lr_schedule.step()
        
        is_best = False
        is_epoch = ((epoch + 1) % args.save_freq) == 0
        if is_epoch:
            print('=> saving checkpoint')
            dist_utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                # 'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                # 'best_acc1': best_metric,
                'args': args,
            }, is_best, args.output_dir, is_epoch=is_epoch)
        
    if dist_utils.is_main_process() and args.wandb:
        wandb_run.finish()
    
        
def train(train_loader, model, optimizer, scaler, epoch, lr_schedule, args):
    iters_per_epoch = len(train_loader) // args.update_freq
    model.train()
    start = time.time()
    total_loss = 0
    total_item = 0
    for data_iter, samples in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq
        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            #if lr_schedule is not None:
            if args.lr_scheduler_type == 'cosine':
                param_group['lr'] = lr_schedule[it]
        
        samples = {k : samples[k].cuda(args.gpu, non_blocking=True) for k in samples}
        
        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            loss = model(
                samples,
                use_checkpoint=args.use_checkpoint,
            )
            loss /= args.update_freq
            
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        
        scaler.scale(loss).backward()
        

        if (data_iter + 1) % args.update_freq != 0:
            continue

        if args.clip_grad_value is not None:
            scaler.unscale_(optimizer)
            if args.clip_grad_type == 'norm':
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_value, norm_type=2.
                )
            elif args.clip_grad_type == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
            else:
                assert False, f"Unknown clip mode ({args.clip_grad_type})."
                
                
        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        
        model.zero_grad(set_to_none=True)
        
        if hasattr(dist_utils.get_model(model), 'logit_scale'):
            # clamp logit scale to [0, 100]
            dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = dist_utils.get_model(model).logit_scale.exp().item()
        else:
            logit_scale = torch.nan

        total_loss += loss.item()
        total_item += len(samples)
        if optim_iter % args.print_freq == 0:
            if dist_utils.is_main_process() and args.wandb:
                wandb.log({"loss": loss})
                
            if dist_utils.is_main_process():
                tr = convert_time(time.time() - start)
                print("Epoch", epoch, ":iter", data_iter, "/", iters_per_epoch, ", loss:", round(loss.item(), 2), 
                      ", running loss:", round(total_loss/total_item, 2), ", time passed:", tr)
                
if __name__ == '__main__':
    parents = defaultConfigs()
    parser = argparse.ArgumentParser('Hierarchical Video Captioning', parents=[parents])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)