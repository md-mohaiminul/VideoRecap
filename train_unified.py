import os
import argparse
import time
import math
import sys
from tqdm import tqdm
import wandb
import copy

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
    crop_size = 224
    transforms_list = [
        Permute([3, 0, 1, 2]),    # T H W C -> C T H W
        transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]),
    ]
    train_transform = transforms.Compose(transforms_list)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    
    all_loaders = []
    #clip caption
    ori_text_feature_type = args.text_feature_type
    if 'clip' in args.unify_type:
        args.metadata = '/checkpoint/mohaiminul/VideoCapHier/datasets/captions_train.pkl'
        args.dataset = 'clip_caption'
        args.video_feature_type = 'pixel'
        args.num_video_feat = 4
        args.video_feature_path = '/checkpoint/kalyanv/ego4d/2022-02-01'
        args.text_feature_type = None
        caption_dataset = VideoCaptionDataset(args, transform = train_transform, is_training=True, force_len = 100000)
        print('len(caption_dataset) = {}'.format(len(caption_dataset)))
        
        collator_caption = CaptionDataCollator(tokenizer, max_gen_tokens = args.max_gen_tokens,
                                        add_bos = True, add_eos = True, pad_token_id = 0)
        
        if args.distributed:
            caption_sampler = torch.utils.data.distributed.DistributedSampler(caption_dataset)
        else:
            train_sampler = None
            
        caption_loader = torch.utils.data.DataLoader(
            caption_dataset, collate_fn=collator_caption, batch_size=args.batch_size, shuffle=(caption_sampler is None),
            num_workers=args.workers, pin_memory=False, sampler=caption_sampler, drop_last=True
        )
        print('len(caption_loader) = {}'.format(len(caption_loader)))
        all_loaders.append(caption_loader)
        

    #segment description
    if 'segment' in args.unify_type:
        args = copy.deepcopy(args)
        args.metadata = '/checkpoint/mohaiminul/VideoCapHier/datasets/clip_summeries_train+pseudo.pkl'
        args.dataset = 'segment_description'
        args.video_feature_type = 'cls'
        args.num_video_feat = 512
        args.video_feature_path = '/checkpoint/mohaiminul/VideoCapHier/datasets/features/cls'
        args.text_feature_type = ori_text_feature_type
        args.num_text_feat = args.num_text_feat
        clip_dataset = VideoCaptionDataset(args, transform = None, is_training=True, force_len = 100000)
        print('len(clip_dataset) = {}'.format(len(clip_dataset)))
        
        if args.distributed:
            clip_sampler = torch.utils.data.distributed.DistributedSampler(clip_dataset)
        else:
            clip_sampler = None
        
        collator_clip = CaptionDataCollator(tokenizer, max_gen_tokens = args.max_gen_tokens,
                                        add_bos = True, add_eos = True, pad_token_id = 0)
            
        clip_loader = torch.utils.data.DataLoader(
            clip_dataset, collate_fn=collator_clip, batch_size=args.batch_size, shuffle=(clip_sampler is None),
            num_workers=args.workers, pin_memory=False, sampler=clip_sampler, drop_last=True
        )
        print('len(clip_loader) = {}'.format(len(clip_loader)))
        all_loaders.append(clip_loader)

    #video summary
    if 'video' in args.unify_type:
        args = copy.deepcopy(args)
        args.metadata = '/checkpoint/mohaiminul/VideoCapHier/datasets/video_summeries_train+pseudo.pkl'
        args.dataset = 'video_summary'
        args.video_feature_type = 'cls'
        args.num_video_feat = 512
        args.video_feature_path = '/checkpoint/mohaiminul/VideoCapHier/datasets/features/video'
        args.text_feature_type = ori_text_feature_type
        args.num_text_feat = args.num_text_feat
        video_dataset = VideoCaptionDataset(args, transform = None, is_training=True, force_len = 100000)
        print('len(video_dataset) = {}'.format(len(video_dataset)))
        
        if args.distributed:
            video_sampler = torch.utils.data.distributed.DistributedSampler(video_dataset)
        else:
            video_sampler = None
            
        collator_video = CaptionDataCollator(tokenizer, max_gen_tokens = args.max_gen_tokens,
                                        add_bos = True, add_eos = True, pad_token_id = 0)
        
        video_loader = torch.utils.data.DataLoader(
            video_dataset, collate_fn=collator_video, batch_size=args.batch_size, shuffle=(video_sampler is None),
            num_workers=args.workers, pin_memory=False, sampler=video_sampler, drop_last=True
        )
        print('len(video_loader) = {}'.format(len(video_loader)))
        all_loaders.append(video_loader)

    #model
    print("=> Creating model")
    # model = VideoRecap(args, use_vision_model=(args.dataset=='clip_caption'))
    model = VideoRecap(args, use_vision_model_forced=True)

    if args.distributed:
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=True
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
    
    latest = os.path.join(args.output_dir, 'checkpoint.pt')
    if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))
    elif args.resume:
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
                
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    cudnn.benchmark = True
    
    if args.lr_scheduler_type == 'cosine':
        lr_schedule = cosine_scheduler(
            args.lr, args.lr_end, args.epochs, len(clip_loader) // args.update_freq,
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
            for loader in all_loaders:
                loader.sampler.set_epoch(epoch)
            # if 'caption' in args.unify_type:
            #     caption_loader.sampler.set_epoch(epoch)
            # if 'clip' in args.unify_type:
            #     clip_loader.sampler.set_epoch(epoch)
            # if 'video' in args.unify_type:
            #     video_loader.sampler.set_epoch(epoch)

        #train(caption_loader, clip_loader, video_loader, model, optimizer, scaler, epoch, lr_schedule, args)
        train(all_loaders, model, optimizer, scaler, epoch, lr_schedule, args)
        
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
    
        
# def train(caption_loader, clip_loader, video_loader, model, optimizer, scaler, epoch, lr_schedule, args):
#     iters_per_epoch = len(clip_loader) // args.update_freq
#     model.train()
#     start = time.time()
#     for data_iter, (captions, clip, video) in enumerate(zip(caption_loader, clip_loader, video_loader)):
#         tt = 0
#         for samples in (captions, clip, video):

def train(all_loaders, model, optimizer, scaler, epoch, lr_schedule, args):
    iters_per_epoch = len(all_loaders[0]) // args.update_freq
    model.train()
    start = time.time()
    loader_types = ['caption', 'clip', 'video']
    for data_iter in range(len(all_loaders[0])):
        for loader_cnt, loader in enumerate(all_loaders):
            samples = next(iter(loader))
            # for k in samples:
            #     print(loader_types[loader_cnt], k, samples[k].shape)
            # continue
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
                #loss /= args.update_freq
                
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)
            
            scaler.scale(loss).backward()
            
            # if (data_iter + 1) % args.update_freq != 0:
            #     continue

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

            if optim_iter % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({"loss": loss})
                    
                if dist_utils.is_main_process():
                    tr = convert_time(time.time() - start)
                    print("Epoch", epoch, ":iter", data_iter, "/", iters_per_epoch, loader_types[loader_cnt], ", loss", loss.item(), ", time passed", tr)
        
if __name__ == '__main__':
    parents = defaultConfigs()
    parser = argparse.ArgumentParser('Hierarchical Video Captioning', parents=[parents])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)