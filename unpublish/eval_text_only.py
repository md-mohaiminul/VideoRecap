import json
import pickle
import torch
import argparse
from collections import OrderedDict
import evaluate
from nlgeval import NLGEval
from tqdm import tqdm
import os.path as osp

import evaluate
from nlgeval import NLGEval

from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.datasets_text_only import TextOnlyDataset, TextDataCollator, GPT2DataCollator

def main(args):
    print('Loading checkpoint from', args.resume)
    ckpt = torch.load(args.resume, map_location='cpu')
    old_args = ckpt['args']
    old_args.part = args.part
    
    print(old_args)

    if 'gpt2' in old_args.model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(old_args.model_name)
        model = GPT2LMHeadModel.from_pretrained(old_args.model_name)
        args.max_positions = model.config.n_positions
        tokenizer.pad_token_id = 0
    elif 't5' in old_args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(f'google/{old_args.model_name}')
        model = AutoModelForSeq2SeqLM.from_pretrained(f'google/{old_args.model_name}')
        args.max_positions = model.config.n_positions
    else:
        raise NotImplementedError

    old_args.metadata = args.metadata
    # old_args.dataset = 'segment_description'
    # old_args.caption_type = 'gt'
    if 'dataset' not in old_args:
        old_args.dataset = args.dataset
    if 'caption_type' not in old_args:
        old_args.caption_type = args.caption_type
        
    val_dataset = TextOnlyDataset(old_args, is_training=False)
    print('len(val_dataset) = {}'.format(len(val_dataset)))
    
    # for i in range(10):
    #     s = val_dataset.__getitem__(i)
    #     print(s.keys())
    #     print(s['input_text'])
    # return
    
    if 'gpt2' in old_args.model_name:
        collator = GPT2DataCollator(tokenizer, max_input_tokens=old_args.max_input_tokens, is_training=False)
    elif 't5' in old_args.model_name:
        collator = TextDataCollator(tokenizer, max_input_tokens=old_args.max_input_tokens)
    else:
        raise NotImplementedError
    
    collator = TextDataCollator(tokenizer, max_input_tokens=old_args.max_input_tokens)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))
    
    # for data_iter, samples in enumerate(val_loader):
    #     print(data_iter, samples['indices'])
    #     print(samples['input_ids'].shape, samples['attention_mask'].shape)
    #     print(samples['input_ids'])
    #     print(samples['attention_mask'])
    #     break
    
    # return

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    model = model.to(args.device)
    model.load_state_dict(state_dict, strict=True)
    print('Loading successful.')
    model.eval()

    references = []
    predictions = []
    
    print('results for', args.metadata)
    
    with torch.no_grad():
        for data_iter, samples in enumerate(tqdm(val_loader)):
            if args.do_sample:
                output = model.generate(
                    samples['input_ids'].to(args.device),
                    attention_mask=samples['attention_mask'].to(args.device),
                    do_sample=True,
                    max_length=old_args.max_output_tokens,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    top_k=0,
                    early_stopping=True,
                )
            else:
                output = model.generate(
                    samples['input_ids'].to(args.device),
                    attention_mask=samples['attention_mask'].to(args.device),
                    do_sample=False,
                    max_new_tokens=old_args.max_output_tokens,
                    early_stopping=True,
                )
            if 'gpt2' in old_args.model_name:
                output = output[:, samples['input_ids'].shape[1]:]
            output = tokenizer.batch_decode(output, skip_special_tokens=True)
            for j in range(len(output)):
                sample = val_dataset.samples[samples['indices'][j].item()]
                if args.infer_only:
                    if old_args.dataset=='segment_description':
                        sample['summary_text'] = output[j].strip()
                    elif old_args.dataset=='video_summary':
                        #print(j, sample['video_summary'])
                        sample['video_summary'] = output[j].strip()
                        #print(j, sample['video_summary'])
                else:
                    if old_args.dataset=='segment_description':
                        references.append(sample['summary_text'].strip().lower())
                    elif old_args.dataset=='video_summary':
                        references.append(sample['video_summary'].strip().lower())
                    predictions.append(output[j].strip().lower())

    if args.infer_only:
        #/checkpoint/mohaiminul/VideoCapHier/datasets/summaries_pseudo_t5l_new.pkl
        #f'/checkpoint/mohaiminul/VideoCapHier/datasets/clip_summaries_pseudo_rest_with_sum.pkl'
        # outfile = f'/checkpoint/mohaiminul/VideoCapHier/datasets/clip_summarries_val_all_v2_with_sum.pkl'
        #outfile = f'/checkpoint/mohaiminul/VideoCapHier/datasets/video_summaries_rest/clip_summaries_pseudo_rest_with_sum.json'
        #outfile = '/checkpoint/mohaiminul/VideoCapHier/datasets/video_summaries_rest/video_summaries_pseudo_new_with_sum.json'
        outfile = f'/checkpoint/mohaiminul/VideoCapHier/datasets/video_summaries_rest/temp/{args.part}.json'
        with open(outfile, 'w') as f:
            json.dump(val_dataset.samples, f, indent=4)
        print('Saved to: ', outfile)
    else:
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
        
        # f = open(osp.join(args.output_dir, f'eval_results_v1_part+v2.txt'), 'w')
        # for k in results:
        #     print('{:16s} = {:9.4f}'.format(k, results[k]))
        #     f.write('{:16s} = {:9.4f} \n'.format(k, results[k]))
        # f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Oracle evaluation', add_help=False)
    
    parser.add_argument('--metadata', type=str, default=None)
    parser.add_argument('--caption_type', type=str, default='gt', choices=['gt', 'lavila', 'blip2'])
    parser.add_argument('--dataset', default='segment_description', type=str, choices=['segment_description', 'video_summary'])
    parser.add_argument('--part', type=int, default=None)
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_output_tokens', type=int, default=77)  #77, 250
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--infer_only', action='store_true')
    
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--top_k', default=None, type=int)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--temperature', default=0.7, type=float) #0.7

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)