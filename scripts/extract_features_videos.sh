CUDA_VISIBLE_DEVICES=1 \
python extract_features_videos.py --output_dir features/temp \
               --video_feature_type pixel --chunk_len -1 --feature_step 4 \
               --video_loader_type moviepy \
               --video_feature_path /data/mmiemon/datasets/ego4d/v1/video_540ss \
               --video_encoder_ckpt /data/mmiemon/LaVila/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth \
               --batch_size 32 --workers 10