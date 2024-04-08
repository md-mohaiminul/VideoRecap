# Video ReCap

[Video ReCap: Recursive Captioning of Hour-Long Videos](https://arxiv.org/abs/2402.13250)\
Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan, Lorenzo Torresani, Gedas Bertasius\
Accepted by **CVPR 2024**\
[[Website](https://sites.google.com/view/vidrecap)] [[Paper](https://arxiv.org/abs/2402.13250)] [[Dataset](https://github.com/md-mohaiminul/VideoRecap/blob/master/datasets.md)] [[Hugging Face](https://huggingface.co/papers/2402.13250)] [[Demo](demo.ipynb)]

ViderReCap is a recursive video captioning model that can process very long videos (e.g., hours long) and output video captions at multiple hierarchy levels: short-range clip captions, mid-range segment descriptions, and long-range video summaries. First, the model generates captions for short video clips of a few seconds. As we move up the hierarchy, the model uses sparsely sampled video features and captions generated at the previous hierarchy level as inputs to produce video captions for the current hierarchy level.

<img src="assets/framework.png"> 


**Code, models, and datasets are temporarily unavailable. We will release it soon.**


