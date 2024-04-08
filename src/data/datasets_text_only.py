import torch
from torch.utils.data import Dataset
import pickle
import copy
import json
import math

def preprocess(caption):
    """Process the question to make it canonical."""
    return caption.strip(" ").strip(".").lower() + "."

class TextOnlyDataset(Dataset):
    def __init__(self, args, is_training = True):
        self.dataset = args.dataset
        self.caption_type = args.caption_type
        
        if 'pkl' in args.metadata:
            with open(args.metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif 'json' in args.metadata:
            with open(args.metadata, 'r') as f:
                self.samples = json.load(f)
        else:
            raise NotImplementedError
        
        if args.part is not None:
            step = math.ceil(len(self.samples)/8)
            print(args.part, args.part*step, (args.part+1)*step)
            self.samples = self.samples[args.part*step : (args.part+1)*step]
        
        self.task_prefix = "Summarize: "
        self.is_training = is_training
        
        if 'gpt2' in args.model_name:
            self.task_suffix = "\nSummary: "

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        input_text = self.task_prefix
        
        if self.dataset=='segment_description':
            if self.caption_type=='gt':
                captions = s['captions_gt']
                #captions = s['narrations']
            elif self.caption_type=='lavila' or self.caption_type=='blip2':
                captions = s['narrations']
            
            for c in captions:
                if self.caption_type=='gt':
                    narration_text = c['narration_text']
                elif self.caption_type=='lavila':
                    narration_text = c[3]
                elif self.caption_type=='blip2':
                    narration_text = c[-1]
                    
                pattern = ['#C', '#c', '#O', '#o', '#Unsure', '#unsure']
                for p in pattern:
                    narration_text = narration_text.replace(p, '')
                narration_text = narration_text.strip('.,\n\r\t\' ') + '. '
                input_text += narration_text
            input_text = input_text.strip()
            
        elif self.dataset=='video_summary':
            if self.caption_type=='gt':
                if isinstance(s['clip_summaries'], str):
                    input_text += s['clip_summaries']
                else:
                    for c in s['clip_summaries']:
                        input_text += c['summary_text']+ ' '
                        
            elif self.caption_type=='lavila':
                if isinstance(s['clip_summaries_pred'], str):
                    input_text += s['clip_summaries_pred']
                else:
                    for c in s['clip_summaries_pred']:
                        input_text += c+ ' '
                
            for p in ['[', ']', ',', '\'']:
                input_text = input_text.replace(p, '')
            input_text = input_text.strip()
        else:
            raise NotImplemented
         
        if hasattr(self, 'task_suffix'):
            input_text += self.task_suffix
        
        sample = {'index': idx, 'input_text': input_text}
        
        if self.is_training:
            if self.dataset=='segment_description':
                sample['output_text'] = s['summary_text'].strip()
            elif self.dataset=='video_summary':
                sample['output_text'] = s['video_summary'].strip()
        return sample


class TextDataCollator:
    def __init__(self, tokenizer, max_input_tokens=512, max_output_tokens=77):
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

    def __call__(self, batch):
        samples = self.tokenizer([x['input_text'] for x in batch], padding = "longest",
                    max_length = self.max_input_tokens, truncation = True, return_tensors = "pt")
        samples['indices'] = torch.tensor([x['index'] for x in batch])
        
        if 'output_text' in batch[0]:
            target_encoding = self.tokenizer([x['output_text'] for x in batch], padding = "longest",
                max_length = self.max_output_tokens, truncation = True, return_tensors = "pt" )
            labels = target_encoding.input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
            samples['labels'] = labels
        
        return samples
    
class GPT2DataCollator:
    def __init__(self, tokenizer, max_input_tokens=512, max_output_tokens=77, is_training=True):
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.pad_token_id = 0
        self.is_training = is_training

    def __call__(self, batch):
        samples = {}
        samples['indices'] = torch.tensor([x['index'] for x in batch])
        
        batch_tokens = []
        input_lengths = []
        for x in batch:
            input_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(x['input_text'][:self.max_input_tokens])
            input_lengths.append(len(input_tokens))
            if 'output_text' in x:
                output_tokens = self.tokenizer.encode(x['output_text'][:self.max_output_tokens]) + [self.tokenizer.eos_token_id]
            batch_tokens.append(input_tokens + output_tokens)
        
        max_len = max([len(x) for x in batch_tokens])
        
        if self.is_training:
            for i in range(len(batch_tokens)):
                padding = max_len - len(batch_tokens[i])
                if padding>0:
                    batch_tokens[i] += [self.pad_token_id for _ in range(padding)]
        else:
            attention_mask = []
            for i in range(len(batch_tokens)):
                padding = max_len - len(batch_tokens[i])
                mask = torch.ones(len(batch_tokens[i]))
                if padding>0:
                    batch_tokens[i] += [self.pad_token_id for _ in range(padding)]
                    mask += torch.zeros(padding)
                attention_mask.append(mask)
            samples['attention_mask'] = torch.tensor(attention_mask)
            
        samples['input_ids'] = torch.tensor(batch_tokens) 
        
        if 'output_text' in batch[0]:
            labels = copy.deepcopy(samples['input_ids'])
            labels[labels == self.pad_token_id] = -100
            for i in range(len(input_lengths)):
                labels[i, :input_lengths[i]] = -100
            samples['labels'] = labels
        
        return samples