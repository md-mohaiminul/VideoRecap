
# if videos are raw Ego4D videos
CUDA_VISIBLE_DEVICES=0 \
python eval_unified.py --resume pretrained_models/videorecap_unified.pt \
        --video_loader_type moviepy --chunk_len -1 \
        --num_video_feat 4 \
        --output_dir outputs/unified \
        --batch_size 32 --workers 10

# if videos are cropped to 288px and chunked to 5 minutes (see scripts/crop_and_resize.sh)
# CUDA_VISIBLE_DEVICES=0 \
# python eval_unified.py --resume pretrained_models/videorecap_unified.pt \
#         --video_loader_type decord --chunk_len 300 \
#         --num_video_feat 4 \
#         --output_dir outputs/unified \
#         --batch_size 32 --workers 10