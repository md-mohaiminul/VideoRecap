import argparse
from collections import OrderedDict
import os.path as osp
import json
import pickle
import os
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

    torch.backends.cudnn.benchmark = True

    #Load checkpoint
    ckpt = torch.load(args.resume, map_location='cpu')
    old_args = ckpt['args']
    old_args.video_feature_type = args.video_feature_type
    old_args.video_feature_path = args.video_feature_path
    old_args.metadata = args.metadata
    if old_args.video_sampling_type=='random':
        old_args.video_sampling_type = 'uniform'
    print(old_args)
    
    # Create and load model
    tokenizer = AutoTokenizer.from_pretrained(old_args.decoder_name)
    collator = CaptionDataCollator(tokenizer, max_gen_tokens = old_args.max_gen_tokens,
                                    add_bos = True, add_eos = True, pad_token_id = 0)

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("=> Creating model")
    model = VideoRecap(old_args)
    model = model.to(args.device)
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    model.eval()
    if old_args.use_half:
        model.half()

    print('Saving to', args.output_dir)
    os.makedirs(args.output_dir, exist_ok = True)

    with open('datasets/videos_train.json', 'r') as f:
        videos = json.load(f)
    with open('datasets/videos_val.json', 'r') as f:
        videos += json.load(f)
    print('Total Videos', len(videos))

    with open(f'{args.metadata}', 'r') as f:
        all_captions = json.load(f)

    all_segments = {}
    total = 0
    for video in videos:
        metadata = []  
        for i in np.arange(0, video['end_sec'], args.feature_step):
            dd = {}
            dd['vid'] = video['vid']
            dd['start_sec'] = i
            dd['end_sec'] = min(i+args.feature_step, video['end_sec'])
            captions = []
            for c in all_captions[video['vid']]:
                if c[1]>=dd['start_sec'] and c[2]<=dd['end_sec']:
                    captions.append(c)
            dd['captions_pred'] = captions
            metadata.append(dd)
            
        old_args.metadata = metadata
        
        dataset = VideoCaptionDataset(old_args, transform=None)
        
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collator, num_workers=args.workers, pin_memory=True, drop_last=False)
        
        captions = {}
        with torch.no_grad():
            for data_iter, samples in enumerate(tqdm(data_loader, desc=f'{total}')):
                indices = samples['indices']
                if hasattr(model, "vision_model"):
                    image = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous().to(args.device)  # BCTHW -> BTCHW
                    samples["video_features"] = model.vision_model.forward_features(image, use_checkpoint=old_args.use_checkpoint, cls_at_last=False)  # NLD
                
                queries = model.map_features(samples)
            
                if old_args.caption_sample == 'multinomial_sample':
                    generated_text_ids, ppls = model.generate(
                        queries,
                        tokenizer,
                        do_sample = False,
                        max_text_length=old_args.max_gen_tokens,
                        num_return_sequences=old_args.caption_num_return_sequences,
                    )
                    
                for j in range(generated_text_ids.shape[0] // args.caption_num_return_sequences):
                    sample = dataset.samples[indices[j].item()]
                    start_sec = sample['start_sec']
                    for k in range(old_args.caption_num_return_sequences):
                        jj = j * old_args.caption_num_return_sequences + k
                        generated_text_str = decode_one(generated_text_ids[jj], tokenizer).strip()
                        captions[start_sec] = generated_text_str

        seconds = list(captions.keys())
        seconds.sort()
        all_segments[video['vid']] = []
        for s in seconds:
            all_segments[video['vid']].append(captions[s])

    with open(f'{args.output_dir}/all_segments.json', 'w') as f:
        json.dump(all_segments, f)
        

if __name__ == '__main__':
    parents = defaultConfigs()
    parser = argparse.ArgumentParser('Hierarchical Video Captioning', parents=[parents])
    parser.add_argument('--caption-sample', default='multinomial_sample',
                        choices=['multinomial_sample', 'beam_sample', 'group_beam_search'])
    parser.add_argument('--feature_step', default=180, type=int, help="Extract one segment description at each 'feature_step' sconds")
    parser.add_argument('--caption-top-k', default=None, type=int)
    parser.add_argument('--caption-top-p', default=0.95, type=float)
    parser.add_argument('--caption-num-beams', default=1, type=int)
    parser.add_argument('--caption-num-beam-groups', default=1, type=int)
    parser.add_argument('--caption-temperature', default=0.7, type=float)
    parser.add_argument('--caption-length-penalty', default=1.0, type=float)
    parser.add_argument('--caption-num-return-sequences', default=1, type=int)
    parser.add_argument('--caption-early-stop', action='store_true', help='early stopping to save computation')
    
    args = parser.parse_args()
    main(args)
