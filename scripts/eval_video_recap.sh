#eval clip captions if videos are raw Ego4D videos
CUDA_VISIBLE_DEVICES=1 \
python eval.py --metadata datasets/clips_val.pkl --eval_freq 100 \
               --output_dir outputs/clips \
               --video_feature_type pixel --video_loader_type moviepy --chunk_len -1 \
               --video_feature_path /data/mmiemon/datasets/ego4d/v1/video_540ss \
               --resume pretrained_models/videorecap/videorecap_clip.pt \
               --batch_size 32 --workers 10

#eval clip captions if videos are cropped to 288px and chunked to 5 minutes (see scripts/crop_and_resize.sh)
# CUDA_VISIBLE_DEVICES=0 \
# python eval.py --metadata datasets/clips_val.pkl --eval_freq 100 \
#                --output_dir outputs/clips \
#                --video_feature_type pixel --video_loader_type decord --chunk_len 300 \
#                --video_feature_path /data/mmiemon/datasets/ego4d/v1/video_540ss \
#                --resume pretrained_models/videorecap/videorecap_clip.pt \
#                --batch_size 32 --workers 10

#eval segment descriptions
CUDA_VISIBLE_DEVICES=1 \
python eval.py --metadata datasets/segments_val.pkl \
               --video_feature_type cls --video_feature_path features/segments \
               --output_dir outputs/segments --resume pretrained_models/videorecap/videorecap_segment.pt \
               --batch_size 32 --workers 10

# eval video summaries
CUDA_VISIBLE_DEVICES=1 \
python eval.py --metadata datasets/videos_val.json \
               --output_dir outputs/videos --resume pretrained_models/videorecap/videorecap_video.pt \
               --video_feature_type cls --video_feature_path features/videos \
               --batch_size 8 --workers 4