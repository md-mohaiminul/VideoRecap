#eval clip captions
CUDA_VISIBLE_DEVICES=0 \
python eval.py --metadata datasets/clips_val.pkl --eval_freq 100 \
               --output_dir outputs/clips \
               --video_feature_type pixel --chunk_len -1 \
               --video_feature_path /data/mmiemon/datasets/ego4d/v1/video_540ss \
               --resume pretrained_models/videorecap/videorecap_clip.pt \
               --video_encoder_ckpt /data/mmiemon/LaVila/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth \
               --batch_size 32 --workers 10

#eval segment descriptions
# video_feature_path=features/segments
# CUDA_VISIBLE_DEVICES=0 \
# python eval.py --metadata datasets/segments_val.pkl \
#                --video_feature_type cls --video_feature_path $video_feature_path \
#                --output_dir outputs/segments --resume pretrained_models/videorecap/videorecap_segment.pt \
#                --batch_size 32 --workers 10

# # eval video summaries
# CUDA_VISIBLE_DEVICES=0 \
# python eval.py --metadata datasets/videos_val.json \
#                --output_dir outputs/videos --resume pretrained_models/videorecap/videorecap_video.pt \
#                --video_feature_type cls --video_feature_path features/videos \
#                --batch_size 8 --workers 4