# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from collections import OrderedDict
import os
import os.path as osp
import pickle
import time
import json
import numpy as np
import csv

import torch
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
    print(args.part)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Available gpu devices", torch.cuda.device_count())

    ckpt = torch.load(args.resume, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    
    if args.video_feature_type is not None:
        old_args.video_feature_type = args.video_feature_type
    old_args.chunk_len = args.chunk_len
    
    #default
    if 'share_mapper' not in old_args:
        old_args.share_mapper = False
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
    if 'video_encoder_ckpt' not in old_args:
        old_args.video_encoder_ckpt = args.video_encoder_ckpt
    if 'freeze_lm_entire' not in old_args:
        old_args.freeze_lm_entire = False
    old_args.video_feature_path = args.video_feature_path
    print(old_args)
    
    #VideoRecap
    # with open('/checkpoint/mohaiminul/VideoCapHier/datasets/clip_summeries_train_es.pkl', 'rb') as f:
    #     clips_sum = pickle.load(f)
    # with open('/checkpoint/mohaiminul/VideoCapHier/datasets/clip_summeries_val_es.pkl', 'rb') as f:
    #     clips_sum += pickle.load(f)
    # print('Summaries', len(clips_sum))
    
    # metadata = []
    # for x in clips_sum:
    #     for c in x['captions_pred']:
    #         metadata.append([c[0], x['sid'], c[1], c[2]])  
    # args.metadata = metadata
    
    f = open('/datasets01/Charades-ego-v1/101320/charades-ego-v1/CharadesEgo/CharadesEgo_v1_test_only1st.csv')
    metadata = []
    step = 1
    csv_reader = csv.reader(f)
    _ = next(csv_reader)  # skip the header
    for cnt, row in enumerate(csv_reader):
        video_id = row[0]
        duration = int(float(row[10]))
        for i in np.arange(0, duration, step):
            metadata.append([video_id, i, min(i+step, duration)])
    print(len(metadata))
    args.metadata = metadata
    
    torch.backends.cudnn.benchmark = True
    
    tokenizer = AutoTokenizer.from_pretrained(old_args.decoder_name)
    
    # Create data
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
    
    #('001d2d1b-d2f9-4c39-810e-6e2087ff9d5a', 0.0, 1.0)
    
    print("=> Creating val dataset")
    old_args.metadata = args.metadata
    
    old_args.part = args.part
    val_dataset = VideoCaptionDataset(old_args, transform=val_transform, 
                                      subsample_stride=args.eval_freq, extract_features=True)
    
    # for i, sample in enumerate(val_dataset):
    #     print(val_dataset.samples[i])
    #     print(sample['index'], sample['video_features'].shape, torch.sum(sample['video_features']))
    #     if i==10:
    #         break
    # return
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    print('len(val_set) = {}'.format(len(val_dataset)))
    print('len(val_loader) = {}'.format(len(val_loader)))

    print("=> Creating model")
    model = VideoRecap(old_args)
    model = model.to(args.device)
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))
    model.eval()
    if args.use_half:
        model.half()
    
    with torch.no_grad():
        for data_iter, samples in enumerate(tqdm(val_loader, desc=f'part_{args.part}')):
            indices = samples['index']
            if hasattr(model, "vision_model"):
                image = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous().to(args.device)  # BCTHW -> BTCHW
                samples["video_features"] = model.vision_model.forward_features(image, use_checkpoint=old_args.use_checkpoint, cls_at_last=False)  # NLD
            
            queries = model.map_features(samples)
        
            if args.caption_sample == 'multinomial_sample':
                generated_text_ids, ppls = model.generate(
                    queries,
                    tokenizer,
                    do_sample = False,
                    max_text_length=args.caption_max_len,
                    num_return_sequences=args.caption_num_return_sequences,
                )
                
            for j in range(generated_text_ids.shape[0] // args.caption_num_return_sequences):
                sample = val_dataset.samples[indices[j]]
                for k in range(args.caption_num_return_sequences):
                    jj = j * args.caption_num_return_sequences + k
                    generated_text_str = decode_one(generated_text_ids[jj], tokenizer).strip()
                    
                if val_dataset.args.dataset == 'clip_caption':
                    val_dataset.samples[indices[j]] = list(sample) + [generated_text_str]
                else:
                    sample['generated_text'] = generated_text_str
                print(data_iter, j, val_dataset.samples[indices[j]])
            break
    
    print(val_dataset.samples[:10])

    outfile = f'/checkpoint/mohaiminul/datasets/charades_ego/captions.pkl'
    print('Saving to ', outfile)
    with open(outfile, 'wb') as f:
        pickle.dump(val_dataset.samples, f)
    

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
    parser.add_argument('--caption-max-len', default=77, type=int)
    # parser.add_argument('--part', default=None, type=int)
    parser.add_argument('--caption-early-stop', action='store_true', help='early stopping to save computation')
    
    args = parser.parse_args()
    
    args.resume = '/checkpoint/mohaiminul/VideoCapHier/outputs/caption/qformer_256/checkpoint.pt'
    # args.metadata = '/checkpoint/mohaiminul/VideoCapHier/datasets/captions_predictions_paceholder_val.pkl'
    args.batch_size = 32
    args.num_workers = 10
    
    main(args)
