import os
import torch
import argparse
from tqdm import tqdm
import time
from collections import OrderedDict

from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM
from transformers import AdamW, get_linear_schedule_with_warmup

import src.utils.distributed as dist_utils
from src.utils.random import random_seed
from src.data.datasets_text_only import TextOnlyDataset, TextDataCollator, GPT2DataCollator

def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def train(args, epoch, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    start = time.time()
    for data_iter, samples in enumerate(train_loader):
        samples = {k : samples[k].cuda(args.gpu, non_blocking=True) for k in samples}
        if 't5' in args.model_name:
            outputs = model(samples['input_ids'], attention_mask=samples['attention_mask'], labels=samples['labels'])
        elif 'gpt2' in args.model_name:
            outputs = model(samples['input_ids'], labels=samples['labels'])
        else:
            raise NotImplementedError
            
        loss = outputs.loss.mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if data_iter % args.print_freq == 0:
            if dist_utils.is_main_process():
                tr = convert_time(time.time() - start)
                print("Epoch", epoch, ":iter", data_iter, "/", len(train_loader), ", loss", loss.item(), ", time passed", tr)

    print("Epoch", epoch, "avg loss:", total_loss / len(train_loader))

def main(args):
    print(args)
    dist_utils.init_distributed_mode(args)
    random_seed(args.seed, dist_utils.get_rank())
    
    print('Saving to', args.output_dir)
    if 'gpt2' in args.model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        model = GPT2LMHeadModel.from_pretrained(args.model_name)

    elif 't5' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(f'google/{args.model_name}')
        model = AutoModelForSeq2SeqLM.from_pretrained(f'google/{args.model_name}')
    else:
        raise NotImplementedError
    print('Model created: ', args.model_name)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total     parameters :', total_params)
    print('Trainable parameters :', trainable_params)
    
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        model.load_state_dict(state_dict, strict=True)
        print('Resumed checkpoint from', args.resume)
    
    train_dataset = TextOnlyDataset(args)
    print('len(train_dataset) = {}'.format(len(train_dataset)))
    
    # for i in range(10):
    #     print(train_dataset[i])
    #     #print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    #     print(train_dataset[i]['input_text'])
    #     #print(tokenizer(train_dataset[0]['input_text']))
    #     print(train_dataset[i]['output_text'])
    # return
    
    if 'gpt2' in args.model_name:
        collator = GPT2DataCollator(tokenizer, max_input_tokens=args.max_input_tokens,
                                max_output_tokens=args.max_output_tokens, is_training = True)
    elif 't5' in args.model_name:
        collator = TextDataCollator(tokenizer, max_input_tokens=args.max_input_tokens, 
                                max_output_tokens=args.max_output_tokens)
    else:
        raise NotImplementedError
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collator, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True
    )
    print('len(train_loader) = {}'.format(len(train_loader)))
    
    # for data_iter, samples in enumerate(train_loader):
    #     print(data_iter, samples['indices'])
    #     print(samples['input_ids'].shape, samples['labels'].shape)
    #     break
    
    # return
    
    if args.distributed:
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=True
        )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_loader)
    )
    
    torch.backends.cudnn.benchmark = True
    
    print("=> beginning training")
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train(args, epoch, model, train_loader, optimizer, scheduler)
        
        is_best = False
        is_epoch = ((epoch + 1) % args.save_freq) == 0
        print('=> saving checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args,
        }, is_best, args.output_dir, is_epoch=is_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Oracle training and evaluation', add_help=False)
    
    # Data
    parser.add_argument('--metadata', type=str, default=None)
    parser.add_argument('--caption_type', type=str, default='gt', choices=['gt', 'lavila', 'blip2'])
    parser.add_argument('--dataset', default='segment_description', type=str, choices=['segment_description', 'video_summary'])
    parser.add_argument('--max_narrations', type=int, default=40) #25
    parser.add_argument('--part', type=int, default=None)
    
    # Model
    parser.add_argument('--model_name', type=str, default='flan-t5-small')
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, required=True, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_input_tokens', type=int, default=512)
    parser.add_argument('--max_output_tokens', type=int, default=77)
    parser.add_argument('--save_freq', default=5, type=int, help='save frequency')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    
    # System
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)