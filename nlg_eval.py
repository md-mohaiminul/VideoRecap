# import evaluate
import json
import pickle
from nlgeval import NLGEval
import evaluate

data = []
for part in range(8):
    with open(f'/checkpoint/mohaiminul/VideoCapHier/outputs/caption/blip2/flan-t5-xl/{part}.json', 'r') as f:
        data += json.load(f)
    print(len(data))

predictions = [x[-1].strip().lower() for x in data]
#references = [x[-2].strip().lower() for x in data]

references = []
pattern = ['#C', '#c', '#O', '#o', '#Unsure', '#unsure']
for x in data:
    narration_text = x[-2]
    for p in pattern:
        narration_text = narration_text.replace(p, '')
    narration_text = narration_text.strip('.,\n\r\t\' ') + '. '
    narration_text = narration_text.strip().lower()
    if narration_text[0]=='c':
        narration_text = 'a person' + narration_text[1:]
    references.append(narration_text)


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

f = open('/checkpoint/mohaiminul/VideoCapHier/outputs/caption/blip2/flan-t5-xl_eval_result.txt', 'w')
for k in results:
    print('{:16s} = {:9.3f}'.format(k, results[k]))
    f.write('{:16s} = {:9.3f} \n'.format(k, results[k]))
f.close()