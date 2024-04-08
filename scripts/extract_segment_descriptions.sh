
video_feature_path=features/videos
metadata=datasets/all_captions.json
feature_step=180 #Extract one segment description at each 180 seconds
CUDA_VISIBLE_DEVICES=0 \
python extract_segment_descriptions.py --metadata $metadata --feature_step $feature_step \
               --video_feature_type cls --video_feature_path $video_feature_path \
               --output_dir datasets --resume pretrained_models/videorecap/videorecap_segment.pt \
               --batch_size 8 --workers 4