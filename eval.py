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
    old_args.video_encoder_ckpt = args.video_encoder_ckpt
    old_args.metadata = args.metadata

    old_args.chunk_len = args.chunk_len
    if old_args.video_sampling_type=='random':
        old_args.video_sampling_type = 'uniform'
    
    print(old_args)
    
    # Create data
    print("=> Creating val dataset")

    tokenizer = AutoTokenizer.from_pretrained(old_args.decoder_name)
    
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

    val_dataset = VideoCaptionDataset(old_args, transform=val_transform, 
                                      subsample_stride=args.eval_freq, is_training=False)

    collator = CaptionDataCollator(tokenizer, max_gen_tokens = old_args.max_gen_tokens,
                                    add_bos = True, add_eos = True, pad_token_id = 0)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=collator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    print('len(val_set) = {}'.format(len(val_dataset)))
    print('len(val_loader) = {}'.format(len(val_loader)))
    print('Saving outputs to', args.output_dir)
    
    # Create and load model
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("=> Creating model")
    model = VideoRecap(old_args)
    model = model.to(args.device)
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    model.eval()
    if args.use_half:
        model.half()

    references = []
    predictions = []
    
    print('results for', args.resume)
    print('Saving to', args.output_dir)
    os.makedirs(args.output_dir, exist_ok = True)
    
    # Eval
    with torch.no_grad():
        for data_iter, samples in enumerate(tqdm(val_loader)):
            indices = samples['indices']
            if hasattr(model, "vision_model"):
                image = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous().to(args.device)  # BCTHW -> BTCHW
                samples["video_features"] = model.vision_model.forward_features(image, use_checkpoint=old_args.use_checkpoint, cls_at_last=False)  # NLD
            
            queries = model.map_features(samples)
        
            if args.caption_sample == 'multinomial_sample':
                generated_text_ids, ppls = model.generate(
                    queries,
                    tokenizer,
                    do_sample = False,
                    max_text_length=old_args.max_gen_tokens,
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
                        references.append(sample['segment_description'].strip().lower())
                    elif val_dataset.args.dataset == 'video_summary':
                        references.append(sample['video_summary'].strip().lower())
                if val_dataset.args.dataset == 'clip_caption':
                    val_dataset.samples[indices[j].item()] = list(sample) + [generated_text_str]
                else:
                    sample['generated_text'] = generated_text_str

    # Save outputs and results
    with open(f"{args.output_dir}/outputs_{old_args.dataset}.json", 'w') as f:
        f.write(json.dumps(val_dataset.samples[:10], indent=4))

    results = {}
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)
    metrics_dict = nlgeval.compute_metrics([references], predictions)
    results['CIDEr'] = metrics_dict['CIDEr']
    results['METEOR'] = metrics_dict['METEOR']

    rouge = evaluate.load('rouge')
    rouge = rouge.compute(predictions=predictions, references=references)
    results['rougeLsum'] = rouge['rougeLsum']
    
    f = open(osp.join(args.output_dir, f"eval_results_{ckpt['epoch']}.txt"), 'w')
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
