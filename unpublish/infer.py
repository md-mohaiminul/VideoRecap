# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from collections import OrderedDict
import os.path as osp
import json

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from transformers import AutoTokenizer
from tqdm import tqdm

import evaluate
from nlgeval import NLGEval

from src.data.video_transforms import Permute
from src.configs.defaults import defaultConfigs
from src.models.video_recap import VideoRecap
from src.data.datasets import VideoCaptionDataset, CaptionDataCollator


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
def decode_one(generated_ids, tokenizer):
    if tokenizer.eos_token_id == tokenizer.bos_token_id:
        if tokenizer.eos_token_id in generated_ids[1:].tolist():
            eos_id = generated_ids[1:].tolist().index(tokenizer.eos_token_id) + 1
        else:
            eos_id = len(generated_ids.tolist()) - 1
    elif tokenizer.eos_token_id in generated_ids.tolist():
        eos_id = generated_ids.tolist().index(tokenizer.eos_token_id)
    else:
        eos_id = len(generated_ids.tolist()) - 1
    generated_text_str = tokenizer.decode(generated_ids[1:eos_id].tolist())
    return generated_text_str

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Available gpu devices", torch.cuda.device_count())

    ckpt = torch.load(args.resume, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    
    if args.video_feature_type is not None:
        old_args.video_feature_type = args.video_feature_type
    if 'chunk_len' not in old_args:
        old_args.chunk_len = args.chunk_len
    
    #default
    if 'finetune_mapper' not in old_args:
        old_args.finetune_mapper = False
    if 'query_width' not in old_args:
        old_args.query_width = args.query_width
    if 'vision_model_type' not in old_args:
        old_args.vision_model_type = args.vision_model_type
    if 'cross_attn_freq' not in old_args:
        old_args.cross_attn_freq = args.cross_attn_freq
    if 'use_lora' not in old_args:
        old_args.use_lora = args.use_lora
    if 'share_mapper' not in old_args:
        old_args.share_mapper = False
    if 'freeze_lm_entire' not in old_args:
        old_args.freeze_lm_entire = False
        
    old_args.video_feature_path = args.video_feature_path
    old_args.dataset = args.dataset
    
    
    print(old_args)

    torch.backends.cudnn.benchmark = True
    
    tokenizer = AutoTokenizer.from_pretrained(old_args.decoder_name)
    
    # Create data
    #if args.video_feature_type=='pixel':
    if old_args.dataset=='clip_caption':
        crop_size = 224
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
        ])
    else:
        val_transform = None
        
    print("=> Creating val dataset")
    old_args.metadata = args.metadata
    
    # if old_args.video_sampling_type=='random':
    #     old_args.video_sampling_type = 'uniform'
    old_args.video_sampling_type = 'uniform'
        
    val_dataset = VideoCaptionDataset(old_args, transform=val_transform, 
                                      subsample_stride=args.eval_freq, is_training=False)
    
    # for i in range(len(val_dataset)):
    #     s = val_dataset[0]
    #     print(i, s['video_features'].shape)
    # return

    # #val_dataset = IndexedDataset(val_dataset)
    # return

    collator = CaptionDataCollator(tokenizer, max_gen_tokens = old_args.max_gen_tokens,
                                    add_bos = True, add_eos = True, pad_token_id = 0)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=collator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    
    # for data_iter, samples in enumerate(tqdm(val_loader)):
    #     print(samples['indices'])
    #     print(samples['video_features'].shape)
    #     print(samples['text_features'].shape)
    #     print(samples['text_mask'].shape)
    #     break
    # return
    
    print('len(val_set) = {}'.format(len(val_dataset)))
    print('len(val_loader) = {}'.format(len(val_loader)))
    print('Saving outputs to', args.output_dir)
    
    print("=> Creating model")
    #model = VideoRecap(old_args, use_vision_model=(old_args.dataset=='clip_caption'))
    model = VideoRecap(old_args)
    model = model.to(args.device)
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    model.eval()
    if args.use_half:
        model.half()
    
    print('results for', args.resume)
    
    with torch.no_grad():
        for data_iter, samples in enumerate(tqdm(val_loader)):
            # indices = indices.tolist()
            indices = samples['indices']
            if hasattr(model, "vision_model"):
                print(samples["video_features"].shape)
                image = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous().to(args.device)  # BCTHW -> BTCHW
                samples["video_features"] = model.vision_model.forward_features(image, use_checkpoint=old_args.use_checkpoint, cls_at_last=False)  # NLD
            
            queries = model.map_features(samples)
        
            if args.caption_sample == 'multinomial_sample':
                generated_text_ids, ppls = model.generate(
                    queries,
                    tokenizer,
                    do_sample = False,
                    # target=None,
                    max_text_length=old_args.max_gen_tokens,
                    #top_k=args.caption_top_k,
                    #top_p=args.caption_top_p,
                    num_return_sequences=args.caption_num_return_sequences,
                    #temperature=args.caption_temperature,
                    #early_stopping=args.caption_early_stop,
                )
                
            for j in range(generated_text_ids.shape[0] // args.caption_num_return_sequences):
                sample = val_dataset.samples[indices[j].item()]
                for k in range(args.caption_num_return_sequences):
                    jj = j * args.caption_num_return_sequences + k
                    generated_text_str = decode_one(generated_text_ids[jj], tokenizer).strip()
                    # ppls_list.append(ppls[jj].item())
                    
                if val_dataset.args.dataset == 'clip_caption':
                    val_dataset.samples[indices[j].item()] = list(sample) + [generated_text_str]
                else:
                    sample['generated_text'] = generated_text_str
            #         print(sample['generated_text'])
            # break

    with open(f"/checkpoint/mohaiminul/datasets/EgoSchema/VideoRecap/captions_all_unique_clip_sum.json", 'w') as f:
        f.write(json.dumps(val_dataset.samples, indent=4))

if __name__ == '__main__':
    parents = defaultConfigs()
    parser = argparse.ArgumentParser('Hierarchical Video Captioning', parents=[parents])
    parser.add_argument('--caption-sample', default='multinomial_sample',
                        choices=['multinomial_sample', 'beam_sample', 'group_beam_search'])
    parser.add_argument('--caption-top-k', default=None, type=int)
    parser.add_argument('--caption-top-p', default=0.95, type=float)
    parser.add_argument('--caption-num-beams', default=1, type=int)
    parser.add_argument('--caption-num-beam-groups', default=1, type=int)
    parser.add_argument('--caption-temperature', default=0.7, type=float)
    parser.add_argument('--caption-length-penalty', default=1.0, type=float)
    parser.add_argument('--caption-num-return-sequences', default=1, type=int)
    parser.add_argument('--caption-early-stop', action='store_true', help='early stopping to save computation')
    
    args = parser.parse_args()
    # args.batch_size = 32
    # args.num_workers = 10
    
    main(args)
