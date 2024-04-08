epochs=5
batch_size=32
workers=10
dataset=clip_caption
metadata=datasets/clips_train.pkl
video_feature_type=pixel
num_video_feat=4 # number of frames
num_video_queries=256
video_feature_path=/data/mmiemon/datasets/ego4d/v1/video_540ss #path to Ego4D videos
video_mapper_type=qformer
chunk_len=300 # crop and divide videos into 5-minute chunks using 'crop_and_resize.sh'
video_loader_type=decord
output_dir=outputs/videorecap/clips

torchrun --nproc_per_node=8 train.py --epochs $epochs \
         --batch_size $batch_size --workers $workers \
         --dataset $dataset --metadata $metadata \
         --video_feature_type $video_feature_type --chunk_len $chunk_len --video_loader_type $video_loader_type \
         --video_feature_path $video_feature_path --num_video_feat $num_video_feat \
         --video_mapper_type $video_mapper_type --num_video_queries $num_video_queries \
         --video_encoder_ckpt pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth \
         --output_dir $output_dir

CUDA_VISIBLE_DEVICES=0 python eval.py \
            --metadata datasets/clips_val.pkl --eval_freq 100 \
            --output_dir $output_dir --resume ${output_dir}