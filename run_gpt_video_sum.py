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

KEYS = ['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',
        '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', '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']
ORGS = ['org-k1Ibm6VYorfXq7bebeiynmw0', 
        'org-k1Ibm6VYorfXq7bebeiynmw0','org-wYCHNi0AlBwxdn35bNzOKXsS']

from openai import OpenAI


pattern = ['#C', '#c', '#O', '#o', '#Unsure', '#unsure']

def get_prompt(captions):
    prompt = 'Here are the captions of a video.\n'
    for c in captions:
        for p in pattern:
            c = c.replace(p, '')
        c = c.strip('.,\n\r\t\' ') + '. '
        prompt += c
    prompt = prompt.strip()
    #prompt += '\nGenerate a one sentence summary of the video.\n'
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
            client = OpenAI(api_key=KEYS[key_idx])
            messages = [{
                "role": "system", 
                "content": "You are a helpful expert in first person view video analysis."
            }]
            messages.append({"role": "user", "content": prompt})  
            res = client.chat.completions.create(model='gpt-3.5-turbo', 
                                                messages=messages, 
                                                temperature=0.0, timeout=10)
            pred = res.choices[0].message.content.strip()
            break
        
        except Exception as e:
            print(e)
            print('Exception trial', trial, 'sleep', 2*(trial+1))
            time.sleep(2*(trial+1))
        
    return pred

def main(args):
    # with open('/checkpoint/mohaiminul/VideoCapHier/outputs/caption/lavila/outputs_stride_1.json', 'r') as f:
    #     narrations = json.load(f)

    # dd = {}
    # for x in narrations:
    #     if x[0] not in dd:
    #         dd[x[0]] = []
    #     dd[x[0]].append(x)
    # narrations = dd

    # dd = {}
    # for v in narrations:
    #     if v not in dd:
    #         dd[v] = []
    #     narrations[v].sort(key=lambda x:x[1])
    #     for x in narrations[v]:
    #         if len(x[-1].split())>30:
    #             continue
    #         dd[v].append(x)
    # narrations = dd
    
    args.metadata = '/checkpoint/mohaiminul/VideoCapHier/datasets/video_summaries_rest/video_summaries_val_v1_part+v2_with_pred.json'
    
    args.output_file = f'/checkpoint/mohaiminul/VideoCapHier/outputs/gpt/{args.model_name}_{args.caption_type}_video_from_clip.json'
    
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
            captions = s['clip_summaries_pred']
            if len(captions)>100:
                indices = np.linspace(0, len(captions)-1, 100).astype(int)
                captions = [captions[i] for i in indices]
            prompt = get_prompt(captions)
            mm.append(len(prompt.split()))
            #print(prompt)
            s['prompt'] = prompt
            pred = get_gpt_response(args.model_name, prompt)
            
            if pred is not None:
                with open(args.output_file, 'w') as f:
                    json.dump(samples, f, indent=4)
                # time.sleep(2)
        else:
            pred = s['generated_text']
            
        if pred is not None: 
            s['generated_text'] = pred
            # print(cnt, s['video_summary'])  
            # print(cnt, s['generated_text'])    
            references.append(s['video_summary'])
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