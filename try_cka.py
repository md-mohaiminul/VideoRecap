import torch
import numpy as np
from cka import CKA, CudaCKA
import time

cka = CudaCKA(device='cuda')

start = time.time()
model = torch.load('pretrained_models/videorecap/videorecap_clip.pt')
model2 = torch.load('pretrained_models/videorecap_unified.pt')

# for (k, v) in model2['state_dict'].items():
#     print(k)
# exit(0)

t1 = []
t2 = []
for (k, v) in model['state_dict'].items():
    if k not in model2['state_dict']:
        continue
    v = v.cuda()
    v2 = model2['state_dict'][k].cuda()
    if torch.sum(v)==torch.sum(v2):
        continue
    try:
        t1.append(cka.linear_CKA(v, v2).item())
        #t2.append(cka.kernel_CKA(v,v2).item())
        print(v.shape)
    except:
        pass

print(sum(t1)/len(t1))

# print(sum(t2)/len(t2))

# for ((k, v),(k2, v2)) in zip (model['state_dict'].items(), model2['state_dict'].items()):
#     if 'crossattention.c_attn.weight' in k:
#         print(v.shape, v2.shape)
#         print(cka.linear_CKA(v, v2), cka.kernel_CKA(v,v2))

# np_cka = CKA()

# X = np.random.randn(100, 10).cuda()
# Y = np.random.randn(100, 10).cuda()

# cka = CudaCKA(device='cuda')

# X = torch.randn(10000, 100).cuda()
# Y = torch.randn(10000, 100).cuda()

# print('Linear CKA, between X and Y: {}'.format(cka.linear_CKA(X, Y)))
# print('Linear CKA, between X and X: {}'.format(cka.linear_CKA(X, X)))

# print('RBF Kernel CKA, between X and Y: {}'.format(cka.kernel_CKA(X, Y)))
# print('RBF Kernel CKA, between X and X: {}'.format(cka.kernel_CKA(X, X)))

# end = time.time()

# print(end-start)