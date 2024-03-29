#Use this videos are raw Ego4D videos
video_feature_path=/data/mmiemon/datasets/ego4d/v1/video_540ss # path to Ego4D videos
video_encoder_ckpt=pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth # path to pretrained video encoder
resume=pretrained_models/videorecap/videorecap_clip.pt
feature_step=4 #Extract one caption at each 4 seconds
CUDA_VISIBLE_DEVICES=0 \
python extract_captions.py --output_dir datasets \
               --video_feature_type pixel --chunk_len -1 --feature_step $feature_step \
               --video_loader_type moviepy \
               --video_feature_path $video_feature_path \
               --video_encoder_ckpt $video_encoder_ckpt \
               --resume $resume \
               --batch_size 8 --workers 4

#Use this if videos are cropped to 288px and chunked to 5 minutes (see scripts/crop_and_resize.sh)
# CUDA_VISIBLE_DEVICES=0 \
# python extract_captions.py --output_dir datasets \
#                --video_feature_type pixel --chunk_len 300 --feature_step $feature_step \
#                --video_loader_type decord \
#                --video_feature_path $video_feature_path \
#                --video_encoder_ckpt $video_encoder_ckpt \
#                --resume $resume \
#                --batch_size 8 --workers 4