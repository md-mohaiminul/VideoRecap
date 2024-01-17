import argparse
from collections import OrderedDict
import os.path as osp
import json
import copy
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
    
    ckpt = torch.load(args.resume, map_location='cpu')
    old_args = ckpt['args']

    old_args.video_encoder_ckpt='/data/mmiemon/LaVila/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth'
    old_args.output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok = True)

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("=> Creating model")
    model = VideoRecap(old_args, use_vision_model_forced=True)
    model = model.to(args.device)
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    model.eval()
    if args.use_half:
        model.half()

    torch.backends.cudnn.benchmark = True
    tokenizer = AutoTokenizer.from_pretrained(old_args.decoder_name)
    
    # Caption
    old_args.metadata = 'datasets/clips_val.pkl'
    old_args.dataset = 'clip_caption'
    old_args.num_video_feat = 4
    old_args.video_feature_type='pixel'
    old_args.chunk_len = -1
    old_args.video_feature_path = '/data/mmiemon/datasets/ego4d/v1/video_540ss'
    old_args.text_feature_type = None
    old_args.max_gen_tokens = 77
    old_args.eval_freq = 100
    
    crop_size = 224
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
    ])
    
    val_dataset = VideoCaptionDataset(old_args, transform=val_transform, 
                                      subsample_stride=old_args.eval_freq, is_training=False)

    collator = CaptionDataCollator(tokenizer, max_gen_tokens = old_args.max_gen_tokens,
                                    add_bos = True, add_eos = True, pad_token_id = 0)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=collator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    
    eval(args, val_dataset, val_loader, model, tokenizer, max_gen_tokens=old_args.max_gen_tokens, 
        dataset_name = old_args.dataset, output_dir = old_args.output_dir)
    
    # Clip Summary
    old_args = copy.deepcopy(old_args)
    old_args.metadata = 'datasets/segments_val.pkl'
    old_args.dataset = 'segment_description'
    old_args.video_feature_type='cls'
    old_args.num_video_feat = 512
    old_args.video_feature_path = '/data/mmiemon/LaVila/datasets/features/vclm_base/cls'
    old_args.max_gen_tokens = 77
    if old_args.hier_type == 'recur':
        old_args.text_feature_type = 'token'
        old_args.num_text_feat = 512
    else:
        old_args.text_feature_type = None
    
    val_transform = None
    
    val_dataset = VideoCaptionDataset(old_args, transform=val_transform, is_training=False)
    

    collator = CaptionDataCollator(tokenizer, max_gen_tokens = old_args.max_gen_tokens,
                                    add_bos = True, add_eos = True, pad_token_id = 0)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=collator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    
    eval(args, val_dataset, val_loader, model, tokenizer, max_gen_tokens=old_args.max_gen_tokens, 
         dataset_name = old_args.dataset, output_dir = old_args.output_dir)

    # Video Summary
    old_args = copy.deepcopy(old_args)
    old_args.metadata = 'datasets/videos_val.json'
    old_args.dataset = 'video_summary'
    old_args.video_feature_type='cls'
    old_args.num_video_feat = 512
    old_args.video_feature_path = 'features/videos'
    old_args.max_gen_tokens = 100
    if old_args.hier_type == 'recur':
        old_args.text_feature_type = 'token'
        old_args.num_text_feat = 512
    else:
        old_args.text_feature_type = None
    
    val_transform = None
    
    val_dataset = VideoCaptionDataset(old_args, transform=val_transform, is_training=False)

    collator = CaptionDataCollator(tokenizer, max_gen_tokens = old_args.max_gen_tokens,
                                    add_bos = True, add_eos = True, pad_token_id = 0)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=collator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    eval(args, val_dataset, val_loader, model, tokenizer, max_gen_tokens=old_args.max_gen_tokens, 
         dataset_name = old_args.dataset, output_dir = old_args.output_dir)

def eval(args, val_dataset, val_loader, model, tokenizer, max_gen_tokens, dataset_name, output_dir):
    references = []
    predictions = []
    
    print('len(val_set) = {}'.format(len(val_dataset)))
    print('len(val_loader) = {}'.format(len(val_loader)))
    print('Saving outputs to', output_dir)
    
    with torch.no_grad():
        for data_iter, samples in enumerate(tqdm(val_loader)):
            indices = samples['indices']
            if len(samples["video_features"].shape) == 5:
                image = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous().to(args.device)  # BCTHW -> BTCHW
                samples["video_features"] = model.vision_model.forward_features(image, cls_at_last=False)  # NLD
            
            queries = model.map_features(samples)
        
            if args.caption_sample == 'multinomial_sample':
                generated_text_ids, ppls = model.generate(
                    queries,
                    tokenizer,
                    do_sample = False,
                    max_text_length=max_gen_tokens,
                    num_return_sequences=args.caption_num_return_sequences,
                )
                
            for j in range(generated_text_ids.shape[0] // args.caption_num_return_sequences):
                sample = val_dataset.samples[indices[j].item()]
                for k in range(args.caption_num_return_sequences):
                    jj = j * args.caption_num_return_sequences + k
                    generated_text_str = decode_one(generated_text_ids[jj], tokenizer).strip()
                predictions.append(generated_text_str.strip().lower())
                
                if args.caption_num_return_sequences == 1:
                    if val_dataset.args.dataset == 'clip_caption':
                        references.append(sample[-1].strip().lower())
                    elif val_dataset.args.dataset == 'segment_description':
                        references.append(sample['summary_text'].strip().lower())
                    elif val_dataset.args.dataset == 'video_summary':
                        references.append(sample['video_summary'].strip().lower())
                    
                if val_dataset.args.dataset == 'clip_caption':
                    val_dataset.samples[indices[j].item()] = list(sample) + [generated_text_str]
                else:
                    sample['generated_text'] = generated_text_str
        
    with open(f"{output_dir}/outputs_{dataset_name}.json", 'w') as f:
        f.write(json.dumps(val_dataset.samples[:10], indent=4))
    
    results = {}
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)
    metrics_dict = nlgeval.compute_metrics([references], predictions)
    results['CIDEr'] = metrics_dict['CIDEr']
    results['METEOR'] = metrics_dict['METEOR']

    rouge = evaluate.load('rouge')
    rouge = rouge.compute(predictions=predictions, references=references)
    results['rougeLsum'] = rouge['rougeLsum']
    
    f = open(osp.join(output_dir, f"eval_results_{dataset_name}.txt"), 'w')
    for k in results:
        print('{:16s} = {:9.4f}'.format(k, results[k]))
        f.write('{:16s} = {:9.4f} \n'.format(k, results[k]))
    f.close()

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
    
    main(args)
