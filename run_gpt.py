import torch
import argparse
import evaluate
from nlgeval import NLGEval
from tqdm import tqdm
import openai
import evaluate
from nlgeval import NLGEval

import random
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import time

# OPENAI_API_KEY = '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'
# openai.api_key = OPENAI_API_KEY

# dmorim,org-k1Ibm6VYorfXq7bebeiynmw0,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
# soar,org-k1Ibm6VYorfXq7bebeiynmw0,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
# brad,org-wYCHNi0AlBwxdn35bNzOKXsS,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

# KEYS = ['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', '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',
#         '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', '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']
# ORGS = ['org-gZItwNX8UrhTWLBRMPrBGoBZ', 'org-k1Ibm6VYorfXq7bebeiynmw0', 
#         'org-k1Ibm6VYorfXq7bebeiynmw0','org-wYCHNi0AlBwxdn35bNzOKXsS']

KEYS = ['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',
        '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', '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']
ORGS = ['org-k1Ibm6VYorfXq7bebeiynmw0', 
        'org-k1Ibm6VYorfXq7bebeiynmw0','org-wYCHNi0AlBwxdn35bNzOKXsS']

from src.data.datasets_text_only import TextOnlyDataset

pattern = ['#C', '#c', '#O', '#o', '#Unsure', '#unsure']

def get_prompt(captions):
    prompt = 'Here are the captions of a video.\n'
    for c in captions:
        for p in pattern:
            c = c.replace(p, '')
        c = c.strip('.,\n\r\t\' ') + '. '
        prompt += c
    prompt = prompt.strip()
    prompt += '\nGenerate a one sentence summary of the video.\n'
    return prompt

# def get_prompt(captions):
#     prompt = 'You are given some language description of a first person view video. The descriptions are sequential and non-overlapping which cover the whole video. Here are the descriptions. \n'
#     for c in captions:
#         for p in pattern:
#             c = c.replace(p, '')
#         c = c.strip('.,\n\r\t\' ') + '. '
#         prompt += c
#     prompt = prompt.strip()
#     prompt += '\nPlease generate a one sentence summary of the video.\n'
#     return prompt

def get_gpt_response(model_name, prompt):
    pred = None
    for trial in range(5):
        try:
            key_idx = random.randint(0, len(KEYS) - 1)
            openai.organization = ORGS[key_idx]
            openai.api_key = KEYS[key_idx]
            if model_name == 'curie':
                res = openai.Completion.create(
                    model="text-curie-001",
                    #model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0
                )
                pred = res["choices"][0]["text"].strip()
                
            elif model_name == 'chatgpt':
                messages = [{
                    "role": "system", 
                    "content": "You are a helpful expert in first person view video analysis."
                }]
                messages.append({"role": "user", "content": prompt})  
                res = openai.ChatCompletion.create(model='gpt-3.5-turbo', 
                                                   messages=messages, 
                                                   temperature=0.0,
                                                   request_timeout=10)
                pred = res['choices'][0]['message']['content'].strip()
            break
        
        except Exception as e:
            print('Exception trial', trial, 'sleep', 2*(trial+1))
            time.sleep(2*(trial+1))
        
    return pred

def main(args):
    if args.caption_type=='gt':
        args.metadata = '/checkpoint/mohaiminul/VideoCapHier/datasets/clip_summeries_val_new.pkl'
    elif args.caption_type=='lavila':
        args.metadata = '/checkpoint/mohaiminul/VideoCapHier/datasets/summaries_lavila_val.pkl'
    elif args.caption_type=='blip2':
        args.metadata = '/checkpoint/mohaiminul/VideoCapHier/datasets/summaries_blip2_val.pkl'
    
    args.output_file = f'/checkpoint/mohaiminul/VideoCapHier/outputs/gpt/{args.model_name}_{args.caption_type}.json'
    
    print(args)
    if os.path.exists(args.output_file):
        args.metadata = args.output_file
        
    if 'pkl' in args.metadata:
        with open(args.metadata, 'rb') as f:
            samples = pickle.load(f)
    elif 'json' in args.metadata:
        with open(args.metadata, 'r') as f:
            samples = json.load(f)
    else:
        raise NotImplementedError
    
    mm = []
    references = []
    predictions = []
    for cnt, s in enumerate(tqdm(samples)):
        if 'generated_text' not in s:
            if args.caption_type=='gt':
                captions = [c['narration_text'] for c in s['captions_gt']]
                
            elif args.caption_type=='lavila' or args.caption_type=='blip2':
                captions = [c[3] for c in s['narrations']]
                
            if len(captions)>100:
                indices = np.linspace(0, len(captions)-1, 100).astype(int)
                captions = [captions[i] for i in indices]
            prompt = get_prompt(captions)
            mm.append(len(prompt.split()))
            s['prompt'] = prompt
            pred = get_gpt_response(args.model_name, prompt)
            
            if pred is not None:
                with open(args.output_file, 'w') as f:
                    json.dump(samples, f, indent=4)
                time.sleep(2)
        else:
            pred = s['generated_text']
            
        if pred is not None: 
            s['generated_text'] = pred
            print(cnt, s['summary_text'])  
            print(cnt, s['generated_text'])    
            references.append(s['summary_text'])
            predictions.append(s['generated_text'])
    
    with open(args.output_file, 'w') as f:
        json.dump(samples, f, indent=4)
                        
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    print(results)

    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)
    metrics_dict = nlgeval.compute_metrics([references], predictions)
    print(metrics_dict)
    for k in metrics_dict:
        results[k] = metrics_dict[k]

    bertscore = evaluate.load("bertscore")
    bertscore = bertscore.compute(predictions=predictions, references=references, lang="en")['f1']
    bert_score = sum(bertscore) / len(bertscore)
    print('Average bertscore', bert_score)
    results['BertScore'] = bert_score
    
    
    results['total_samples'] = len(samples)
    results['valid'] = len(predictions)
    
    print(results)
    
    with open(args.output_file.split('.')[0]+'_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT evaluation', add_help=False)
    
    parser.add_argument('--caption_type', type=str, default='lavila', choices=['gt', 'lavila', 'blip2'])
    parser.add_argument('--dataset', default='segment_description', type=str, choices=['segment_description', 'video_summary'])
    parser.add_argument('--part', default=None, type=int)
    parser.add_argument('--model_name', default='chatgpt', type=str, choices=['curie', 'chatgpt'])
    
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--top_k', default=None, type=int)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--temperature', default=0.7, type=float) #0.7

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)