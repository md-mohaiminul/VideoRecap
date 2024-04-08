# Train videorecap with the manually annotated data
# We initialize from pretrained videorecap_clip checkpoint and train for 10 epoch.

epochs=15   # Start epoch 5, so train 10 epoch
batch_size=4
dataset=segment_description
metadata=datasets/segments_train.pkl

video_feature_type=cls
num_video_feat=80   #40
num_video_queries=32
video_feature_path=features/segments #path to segments feature
video_mapper_type=qformer
video_sampling_type=uniform

text_feature_type=token
num_text_feat=512
num_text_queries=256
text_mapper_type=qformer

output_dir=outputs/videorecap_80/segments
resume=pretrained_models/videorecap/videorecap_clip.pt

torchrun --nproc_per_node=8 train.py \
         --epochs $epochs --batch_size $batch_size \
         --dataset $dataset --metadata $metadata \
         --output_dir $output_dir --resume $resume \
         --video_feature_path $video_feature_path --num_video_feat $num_video_feat \
         --video_feature_type $video_feature_type \
         --video_mapper_type $video_mapper_type --num_video_queries $num_video_queries \
         --text_feature_type $text_feature_type --num_text_feat $num_text_feat \
         --text_mapper_type $text_mapper_type --num_text_queries $num_text_queries \

CUDA_VISIBLE_DEVICES=0 \
python eval.py --metadata datasets/segments_val.pkl \
               --output_dir $output_dir --resume ${output_dir}/checkpoint.pt \
               --video_feature_type $video_feature_type --video_feature_path $video_feature_path \
               --batch_size 32 --workers 10
