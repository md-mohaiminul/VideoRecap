# VideoRecap

This is the official implementation of our paper [Video ReCap: Recursive Captioning of Hour-Long Videos](link). ViderReCap is a recursive video captioning model that can process very long videos (e.g., hours long) and output video captions at multiple hierarchy levels: short-range clip captions, mid-range segment descriptions, and long-range video summaries. First, the model generates captions for short video clips of a few seconds. As we move up the hierarchy, the model uses sparsely sampled video features and captions generated at the previous hierarchy level as inputs to produce video captions for the current hierarchy level.

<img src="assets/framework.png"> 

## Installation
See [installation.md](installation.md) to install this code.

## Ego4D videos
1. Get [License Agreement](https://ego4d-data.org/docs/start-here/#cli-download) and [download](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) the Ego4D videos. 
2. Use [crop_and_resize.sh](scripts/crop_and_resize.sh) to crop and chunk the videos to the smaller side of 288 pixel and chunk length of 5 minutes. \
(Note: This step is required for faster I/O. You can also evaluate the pretrained models without this step.)

## Ego4D-HCap Dataset
See [datasets.md](datasets.md) to download and prepare datasets.

## Download or extract features
We utilize the video encoder of pretrained Dual-Encoder from [LaViLa](https://github.com/facebookresearch/LaViLa/blob/main/docs/MODEL_ZOO.md) to extract features. \
**You can directly download the extracted features (~30 GB) from [this link]()**. \
Alternatively, you may extract the features on your own using the following steps.

1. Download the pretrained video encoder using the following command.
```bash
wget https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.baseline.ep_0003.pth
```
2. Extract segment features.
```bash
bash scripts/extract_features_segments.sh
```
3. Extract video features.
```bash
bash scripts/extract_features_videos.sh
```

## Train VideoReCap Model

VideoReCap is a recursive model for hierarchical video captioning that uses captions generated at the previous level as input for the current hierarchy. We train VideoReCap utilizing the following curriculum learning strategy.

1. First, train for 5 epochs using the clip captions data.
```bash
bash scripts/run_videorecap_clip.sh
```
2. Then extract captions at each 4 seconds interval for the whole video using the trained clip captioning model of step 1. Replace the 'captions_pred' of the train and val metadata using generated captions from appropriate time windows (See [datasets.md](datasets.md) for more details).
```bash
bash scripts/extract_captions.sh
```
3. Initialize from VideoReCap clip checkpoint and train for 10 epochs using the segment descriptions.
```bash
bash scripts/run_videorecap_segment.sh
```
4. Extract segment descriptions using the at each 180 seconds interval for the whole video using the trained clip captioning model of step 3. Replace the 'segment_descriptions_pred' of the train and val metadata using generated descriptions from appropriate time windows (See [datasets.md](datasets.md) for more details).
```bash
bash scripts/extract_segment_descriptions.sh
```
5. Finally, initialize from VideoReCap segment checkpoint and train for 10 epochs using the video summaries.
```bash
bash scripts/run_videorecap_video.sh
```

## Train VideoReCap-U Model

While VideoReCap trains three different sets of trainable parameters for three hierarchies, VideoReCap-U trains only one set of trainable parameters. Following curriculum learning scheme with an alternate batching technique allows us to train a unified model and avoid catestrophic foregetting.

1. First stage is same as the VideRecap model, where we train for 5 epochs using the clip captions data.
```bash
bash scripts/run_videorecap_clip.sh
```
2. Then extract captions at each 4 seconds interval for the whole video using the trained clip captioning model of step 1. Replace the 'captions_pred' of the train and val metadata using generated captions from appropriate time windows (See [datasets.md](datasets.md) for more details).
```bash
bash scripts/extract_captions.sh
```
3. Secondly, we initialize from VideoReCap clip checkpoint and train for 10 epochs using the segment descriptions and some clip captions data. We sample clip captions and segment descriptions alternatively at each bach. 
```bash
bash scripts/run_videorecap_clip.sh
```
4. Extract segment descriptions using the at each 180 seconds interval for the whole video using the trained clip captioning model of step 3. 
```bash
bash scripts/extract_segment_descriptions.sh
```
5. Finally, we initialize from VideoReCap segment checkpoint and train for 10 epochs using the video summaries and some segment descriptions and clip captions data. We sample data from three hierarchies alternatively at each batch.
```bash
bash scripts/run_videorecap_clip.sh
```

## Evaluate Pretrained Models

We provide our best model for both VideoReCap and VideoReCap-U. \
**Download the pretrained models from [this link]()**
1. Evaluate VideoReCap.
```bash
bash scripts/eval_video_recap.sh
```
2. Evaluate VideoReCap-U.
```bash
bash scripts/eval_video_recap_u.sh
```

You should get the following numbers.

| Model | Clip Caption<br>(C/ R/ M) | Segment Description<br>(C/ R/ M) | Video Summary<br>(C/ R/ M) | Checkpoint |
| --- | --- | --- | --- | --- |
VideoReCap | 98.35/ 48.77/ 28.28 | 46.88/ 39.73/ 18.55 | 29.34/ 32.64/ 14.45 | [download](link)
VideoReCap-U | 92.67/ 47.90/ 28.08 | 45.60/ 39.33/ 18.17 | 31.06/ 33.32/ 14.16 | [download](link)
