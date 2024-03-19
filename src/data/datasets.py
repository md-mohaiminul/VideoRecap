import numpy as np
import pickle
import json
import random
import torch
import decord
import os.path as osp
import math
import os
import h5py

from moviepy.editor import *

from transformers import AutoTokenizer


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def video_loader_decord(root, vid, second, end_second=None, chunk_len=300, fps=30, clip_length=32, jitter=False, return_tensors="pt"):
    if chunk_len == -1:
        vr = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid)))
        second_offset = second
        if end_second is not None:
            end_second = min(end_second, len(vr) / vr.get_avg_fps())
        else:
            end_second = len(vr) / vr.get_avg_fps()
    else:
        chunk_start = int(second) // chunk_len * chunk_len
        second_offset = second - chunk_start
        vr = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start)))
    if fps == -1:
        fps = vr.get_avg_fps()

    # calculate frame_ids
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    if chunk_len == -1:
        if end_second <= second:
            raise ValueError("end_second should be greater than second")
        else:
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
    else:
        frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)
    
    # load frames
    if max(frame_ids) < len(vr):
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    else:
        if chunk_len==-1:
            frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
            frames = vr.get_batch(frame_ids).asnumpy()
        else:
            # find the remaining frames in the next chunk
            try:
                frame_ids_part1 = list(filter(lambda frame_id: frame_id < len(vr), frame_ids))
                frames_part1 = vr.get_batch(frame_ids_part1).asnumpy()
                vr2 = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start + chunk_len)))
                frame_ids_part2 = list(filter(lambda frame_id: frame_id >= len(vr), frame_ids))
                frame_ids_part2 = [min(frame_id % len(vr), len(vr2) - 1) for frame_id in frame_ids_part2]
                frames_part2 = vr2.get_batch(frame_ids_part2).asnumpy()
                frames = np.concatenate([frames_part1, frames_part2], axis=0)
            # the next chunk does not exist; the current chunk is the last one
            except (RuntimeError, decord.DECORDError) as error:
                print(error)
                frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
                frames = vr.get_batch(frame_ids).asnumpy()
            
    if return_tensors=='pt':
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
        return torch.stack(frames, dim=0)
    else:
        return frames
    
def video_loader_moviepy(root, vid, second, end_second=None, chunk_len=-1, fps=30, clip_length=32, jitter=False):
    if chunk_len == -1:
        video_file = osp.join(root, '{}.mp4'.format(vid))
        clip = VideoFileClip(video_file)
        second_offset = second
        if end_second is not None:
            end_second = min(end_second, clip.duration)
        else:
            end_second = clip.duration
    else:
        chunk_start = int(second) // chunk_len * chunk_len
        second_offset = second - chunk_start
        clip = VideoFileClip(osp.join(root, '{}'.format(vid), '{}.mp4'.format(chunk_start)))
    if fps == -1:
        fps = clip.fps
    video_len = int(clip.duration * fps)

    # calculate frame_ids
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    if chunk_len == -1:
        if end_second <= second:
            raise ValueError("end_second should be greater than second")
        else:
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, video_len), num_segments=clip_length, jitter=jitter)
    else:
        frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)

    # load frames
    frames = []
    if max(frame_ids) < video_len:
        #frames = vr.get_batch(frame_ids).asnumpy()
        for i in frame_ids:
            #print(i/fps, clip.duration)
            frames.append(clip.get_frame(i/fps))
    else:
        # find the remaining frames in the next chunk
        try:
            frame_ids_part1 = list(filter(lambda frame_id: frame_id < video_len, frame_ids))
            for i in frame_ids_part1:
                frames.append(clip.get_frame(i / fps))
            #vr2 = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start + chunk_len)))
            clip2 = VideoFileClip(osp.join(root, '{}'.format(vid), '{}.mp4'.format(chunk_start)))
            frame_ids_part2 = list(filter(lambda frame_id: frame_id >= video_len, frame_ids))
            frame_ids_part2 = [min(frame_id % video_len, int(clip2.duration * fps) - 1) for frame_id in frame_ids_part2]
            for i in frame_ids_part2:
                frames.append(clip2.get_frame(i / fps))
            #frames = np.concatenate([frames_part1, frames_part2], axis=0)
        # the next chunk does not exist; the current chunk is the last one
        except (RuntimeError, decord.DECORDError) as error:
            print(error)
            frame_ids = get_frame_ids(min(frame_offset, video_len - 1), video_len, num_segments=clip_length, jitter=jitter)
            for i in frame_ids:
                frames.append(clip.get_frame(i / fps))

    frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    return torch.stack(frames, dim=0)

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq

def sample_features(features, sampling_type, num_feat):
    if sampling_type=='uniform':
        video_mask = torch.ones(num_feat, dtype=torch.long)
        step = features.shape[0] / (num_feat + 1)
        idx = np.arange(0, features.shape[0], step).astype(int)[1:]
        features = features[idx]
        return features

    elif sampling_type=='masking':
        video_mask = torch.zeros(num_feat, dtype=torch.long)
        video_mask[:features.shape[0]] = 1
        if features.shape[0] > num_feat:
            step = features.shape[0] / (num_feat + 1)
            idx = np.arange(0, features.shape[0], step).astype(int)[1:]
            features = features[idx]
        else:
            features2 = np.zeros((num_feat - features.shape[0], features.shape[1]), dtype=np.float16)
            features = np.concatenate((features, features2), axis=0)
        features = features[:num_feat]

    elif sampling_type=='mean':
        video_mask = torch.ones(num_feat, dtype=torch.long)
        step = features.shape[0] / (num_feat + 1)
        idx = np.arange(0, features.shape[0], step).astype(int)
        idx[-1] = features.shape[0]
        features = []
        for i in range(1, len(idx)):
            start = idx[i-1]
            end = idx[i]
            if start==end:
                end = end+1
            x = np.mean(features[start:end], axis=0)
            features.append(x)
        features = np.asarray(features)
        features = features[:num_feat]
        
    elif sampling_type=='random':
        # video_mask = torch.ones(num_feat, dtype=torch.long)
        pop = [i for i in range(features.shape[0])]
        idx = random.choices(pop, k=num_feat)
        idx = sorted(idx)
        features = features[idx]
        return features
    else:
        raise NotImplementedError
    
    return features, video_mask

class VideoCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform=None, is_training = True, subsample_stride=1, 
                 extract_features = False, force_len = None):
        self.args = args
        self.transform = transform
        self.is_training = is_training
        self.force_len = force_len
        
        if isinstance(args.metadata, str):
            if 'pkl' in args.metadata:
                with open(args.metadata, 'rb') as f:
                    self.samples = pickle.load(f)
            elif 'json' in args.metadata:
                with open(args.metadata, 'r') as f:
                    self.samples = json.load(f)
        else:
            self.samples = args.metadata
            
        if subsample_stride>1:
            self.samples = self.samples[::subsample_stride]
        
        self.extract_features = extract_features

    def __getitem__(self, i):
        if self.force_len is not None:
            i = random.randint(0, len(self.samples)-1)
            
        sample = {"index": i}
        
        if self.args.dataset == 'clip_caption':
            if len(self.samples[i])==4:
                if self.extract_features:
                    vid, sid, start_second, end_second = self.samples[i]
                else:
                    vid, start_second, end_second, caption = self.samples[i]
                    sample['caption'] = caption
            elif len(self.samples[i])==3:
                vid, start_second, end_second = self.samples[i]
            try:
                if self.args.video_loader_type=='decord':
                    video_features = video_loader_decord(self.args.video_feature_path, vid, start_second, end_second, 
                                    clip_length=self.args.num_video_feat, chunk_len=self.args.chunk_len)
                elif self.args.video_loader_type=='moviepy':
                    video_features = video_loader_moviepy(self.args.video_feature_path, vid, start_second, end_second, 
                                    clip_length=self.args.num_video_feat, chunk_len=self.args.chunk_len)
                else:
                    raise NotImplementedError
            except:
                print('Error loading', vid, start_second, end_second)
                video_features = torch.zeros([self.args.num_video_feat, 288, 384, 3])
            
        elif self.args.dataset == 'segment_description':
            s = self.samples[i]
            if 'sid' in s:
                sid = s['sid']
                video_features = np.load(f'{self.args.video_feature_path}/{sid}.npy')
            else:
                video_features = np.load(f"{self.args.video_feature_path}/{s['vid']}.npy")
                start = int(s['start_sec']/4) if 'start_sec' in s else 0
                end = int(s['end_sec']//4) if 'end_sec' in s else video_features.shape[0]
                video_features = video_features[start:end+1]
                
            if self.args.video_sampling_type=='uniform':
                video_features = sample_features(video_features, self.args.video_sampling_type, self.args.num_video_feat)
            else:
                video_features, video_mask = sample_features(video_features, self.args.video_sampling_type, self.args.num_video_feat)
                sample['video_mask'] = video_mask
            if 'segment_description' in s:
                sample['caption'] = s['segment_description']
            
        elif self.args.dataset == 'video_summary':
            s = self.samples[i]
            if self.args.video_feature_type=='cls':
                if 'end_sec' not in s:
                    video_features = []
                    for c in self.samples[i]['sids']:
                        x =  np.load(f'features/segments/{c}.npy')
                        video_features.append(x)
                    video_features = np.concatenate(video_features, axis=0)
                else:
                    video_features = np.load(f"{self.args.video_feature_path}/{s['vid']}.npy")
                    start = int(s['start_sec']/4) if 'start_sec' in s else 0
                    end = int(s['end_sec']//4) if 'end_sec' in s else video_features.shape[0]
                    video_features = video_features[start:end+1]
                
                if self.args.video_sampling_type=='uniform':
                    video_features = sample_features(video_features, self.args.video_sampling_type, self.args.num_video_feat)
                else:
                    video_features, video_mask = sample_features(video_features, self.args.video_sampling_type, self.args.num_video_feat)
                    sample['video_mask'] = video_mask  
            else:
                video_features = None
            
            sample['caption'] = self.samples[i]['video_summary']
                   
        else:
            raise NotImplementedError
        
        if self.transform is not None:
            video_features = self.transform(video_features)
        
        if not (video_features is None):
            sample['video_features'] = video_features
        
        if self.args.text_feature_type == 'token':
            if self.args.dataset == 'segment_description':
                sample['text_features'] = ""
                for c in self.samples[i]["captions_pred"]:
                    sample['text_features'] += c[3]
            elif self.args.dataset == 'video_summary':
                if 'version' in self.samples[i] and self.samples[i]['version']=='v1':
                    text_features = self.samples[i]['segment_descriptions_pred']
                    for p in ['[', ']', ',', '\'']:
                        text_features = text_features.replace(p, '')
                    sample['text_features'] = text_features.strip()
                else:
                    text_features = ""
                    for c in self.samples[i]['segment_descriptions_pred']:
                        text_features += c.strip('.').strip() + '.' + ' '
                    sample['text_features'] = text_features.strip()
            else:
                raise NotImplementedError
                
        return sample
        
    def __len__(self):
        if self.force_len is not None:
            return self.force_len
        return len(self.samples)
    
    
class CaptionDataCollator:
    def __init__(self, tokenizer, max_gen_tokens = 77, add_bos = True, add_eos = True, pad_token_id = 0):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.max_gen_tokens = max_gen_tokens
        self.tokenizer.pad_token = pad_token_id

    def __call__(self, batch):
        samples = {}
        samples['indices'] = torch.tensor([x['index'] for x in batch])
        
        if 'video_features' in batch[0]:
            video_features = [x['video_features'] for x in batch]
            samples["video_features"] = torch.from_numpy(np.stack(video_features))
        
        if 'video_mask' in batch[0]:
            video_mask = [x['video_mask'] for x in batch]
            samples["video_mask"] = torch.from_numpy(np.stack(video_mask))
        
        if 'caption' in batch[0]:
            captions = [x['caption'] for x in batch]
            caption_tokens = self.tokenizer(captions)['input_ids']
            
            lengths = [len(c) for c in caption_tokens]
            max_len = min(max(lengths), self.max_gen_tokens)
            
            if self.add_bos:
                max_len += 1
            if self.add_eos:
                max_len += 1
            
            batch_tokens = []
            for tokens in caption_tokens:
                if self.add_bos:
                    tokens = [self.tokenizer.bos_token_id] + tokens
                if self.add_eos:
                    tokens = tokens + [self.tokenizer.eos_token_id]
                    
                padding = max_len - len(tokens)
                if padding>0:
                    tokens += [self.pad_token_id for _ in range(padding)]
                elif padding<0:
                    tokens = tokens[:max_len]
                batch_tokens.append(tokens)
                
            samples["caption_tokens"] = torch.tensor(batch_tokens)
        
        if "text_features" in batch[0]:
            texts = [x['text_features'] for x in batch]
            dd = self.tokenizer(texts, return_tensors="pt", padding="longest", max_length=512, truncation=True)
            samples["text_features"] = dd['input_ids']
            samples["text_mask"] = dd['attention_mask']
        
        return samples