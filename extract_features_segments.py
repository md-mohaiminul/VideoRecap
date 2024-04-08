import json
import torch
from torch import nn
import argparse
from collections import OrderedDict
from tqdm import tqdm
import decord
import os
import numpy as np
import pickle
import math

import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from src.data.video_transforms import Permute
from src.models.model_utils import rsetattr, remap_keys
from src.models.openai_model import QuickGELU, Transformer
from src.models.openai_clip import load as load_openai_clip
from src.models.timesformer import SpaceTimeTransformer
from src.data.datasets import VideoCaptionDataset


def main(args):
    print(args)

    print('Saving to', args.output_dir)
    os.makedirs(args.output_dir, exist_ok = True)

    with open('datasets/segments_train.pkl', 'rb') as f:
        clips_sum = pickle.load(f)
    with open('datasets/segments_val.pkl', 'rb') as f:
        clips_sum += pickle.load(f)
    print('Total Segments', len(clips_sum))

    metadata = []
    for x in clips_sum:
        # Extract 1 feature at each '4' seconds. However, we extracted at least 45 features from each segment.
        step = min((x['end_sec']-x['start_sec'])/45, args.feature_step)
        for i in np.arange(x['start_sec'], x['end_sec'], step):
            metadata.append([x['vid'], x['sid'], i, min(i+step, x['end_sec'])])
    args.metadata = metadata

    print('Total features', len(metadata))
    
    crop_size = 224
    transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]),
        ])
    
    dataset = VideoCaptionDataset(args, transform=transform, extract_features=True)
    
    #Check is video is loading correctly
    # for i in range(10):
    #     sample = dataset[i]
    #     print(dataset.samples[i])
    #     print(sample['index'], sample['video_features'].shape, torch.sum(sample['video_features']))
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                                              num_workers=args.workers, pin_memory=True, drop_last=False)

    print('len(dataset) = {}'.format(len(dataset)))
    print('len(data_loader) = {}'.format(len(data_loader)))
    
    vision_model = SpaceTimeTransformer(
        num_frames=4,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=False,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    
    #freeze visual encoder
    for n, p in vision_model.named_parameters():
        p.requires_grad = False
    
    #load contrastive VLP pretrained weights
    checkpoint = torch.load(args.video_encoder_ckpt, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'visual' in k:
            state_dict[k.replace('module.visual.', '')] = v
    vision_model.load_state_dict(state_dict, strict=True) 
    model = vision_model
    print('Loaded checkpoint from', args.video_encoder_ckpt)
    
    model.eval()
    model.to(args.device)
    
    all_features = {}
    with torch.no_grad():
        for data_iter, samples in enumerate(tqdm(data_loader)):
            image = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous().to(args.device)  # BCTHW -> BTCHW
            features = model.forward_features(image, cls_at_last=True)  # NLD
            for j in range(features.shape[0]):
                sid = dataset.samples[samples['index'][j].item()][1]
                start_sec = dataset.samples[samples['index'][j].item()][2]
                if sid not in all_features:
                    all_features[sid] = {}
                all_features[sid][start_sec] = features[j].detach().cpu().numpy()
    
    for sid in all_features:
        seconds = list(all_features[sid].keys())
        seconds.sort()
        features = []
        for s in seconds:
            features.append(all_features[sid][s])
        features = np.stack(features)
        print(sid, features.shape, np.sum(features))
        np.save(f'{args.output_dir}/{sid}.npy', features)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features', add_help=False)
    parser.add_argument('--dataset', default='clip_caption', type=str)
    parser.add_argument('--video_feature_path', default=None, type=str, required=True)
    parser.add_argument('--video_encoder_ckpt', default=None, type=str, required=True)
    parser.add_argument('--output_dir', default=None, type=str, required=True)
    parser.add_argument('--num_video_feat', default=4, type=int, help='Number of video frames for the video encoder.')
    parser.add_argument('--feature_step', default=4, type=int, help="Extract one feature at each 'feature_step' sconds")
    parser.add_argument('--video_feature_type', default='pixel', type=str)
    parser.add_argument('--text_feature_type', default=None, type=str)
    parser.add_argument('--video_loader_type', default='decord', choices=['decord', 'moviepy'], type=str)
    parser.add_argument('--chunk_len', default=-1, type=int)   #-1/ 300
    parser.add_argument('--part', default=None, type=str)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=10, type=int)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)