video_feature_path=/data/mmiemon/datasets/ego4d/v1/video_540ss # path to Ego4D videos
video_encoder_ckpt=pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth # path to pretrained video encoder
CUDA_VISIBLE_DEVICES=1 \
python extract_features_segments.py --output_dir features/temp \
               --video_feature_type pixel --chunk_len -1 --feature_step 4 \
               --video_loader_type moviepy \
               --video_feature_path  $video_feature_path \
               --video_encoder_ckpt  $video_encoder_ckpt \
               --batch_size 32 --workers 10