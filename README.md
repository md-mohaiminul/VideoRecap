# VideoRecap

## Installation
See [installation.md](installation.md) to install this code.

## Ego4D-Hcap Dataset

See [datasets.md](datasets.md) to download and prepare datasets.

## Download videos and features

## Eval Pretrained Models

1. Evaluate Clip Caption.
```
CUDA_VISIBLE_DEVICES=0 \
python eval.py --metadata datasets/clip_summeries_val.pkl \
               --output_dir outputs/clip_summary \
               --resume pretrained_models/clip_sum_ckpt.pt \
               --video_feature_path features/cls \
               --batch_size 32 --workers 10
```

## Train and Eval VideoRecap Model

## Train and Eval VideoRecap-U Model