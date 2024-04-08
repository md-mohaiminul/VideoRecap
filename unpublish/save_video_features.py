import pickle

ann_file = f'/checkpoint/mohaiminul/VideoCapHier/datasets/video_summaries_train_dict.pkl'
with open(ann_file, 'rb') as f:
    ann = pickle.load(f)
print(len(ann))
sum_lens = {}
sids = []
for v in ann:
    for p in ann[v]:
        for s in ann[v][p]:
            sids.append(s['sid'])
        # if v in sum_lens:
        #     assert sum_lens[v]==len(ann[v][p])
        # else:
        #     sum_lens[v] = len(ann[v][p])
print(len(set(sids)))

ann_file = f'/checkpoint/mohaiminul/VideoCapHier/datasets/video_summaries_train_annotated.pkl'
with open(ann_file, 'rb') as f:
    ann = pickle.load(f)
print(len(ann))

cnt = 0
for v in ann:
    for s in v['sids']:
        if s not in sids:
            cnt += 1
print(cnt)
