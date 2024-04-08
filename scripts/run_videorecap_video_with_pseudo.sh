# Train videorecap with the manually annotated data + LLM-generated pseudo-annotation
# We initialize from pretrained videorecap_clip checkpoint and train for 10 epoch.

epochs=25 # Start epoch 15, so train 10 epoch
batch_size=8
max_gen_tokens=120
dataset=video_summary
metadata=datasets/videos_train.json # manually annotated data
metadata_pseudo=datasets/videos_train_pseudo.json # LLM-generated pseudo-annotation

video_feature_type=cls
num_video_feat=512
video_sampling_type=masking
num_video_queries=256
video_mapper_type=qformer
video_feature_path=features/videos # path to video features

text_feature_type=token
num_text_feat=512
num_text_queries=256
text_mapper_type=qformer

output_dir=outputs/videorecap/videos_pseudo
resume=pretrained_models/videorecap/videorecap_segment.pt

torchrun --nproc_per_node=8 train.py \
         --epochs $epochs --batch_size $batch_size \
         --dataset $dataset --metadata $metadata --metadata_pseudo $metadata_pseudo \
         --video_feature_path $video_feature_path \
         --video_feature_type $video_feature_type --num_video_feat $num_video_feat \
         --video_mapper_type $video_mapper_type --num_video_queries $num_video_queries \
         --video_sampling_type $video_sampling_type \
         --text_feature_type $text_feature_type --num_text_feat $num_text_feat \
         --text_mapper_type $text_mapper_type --num_text_queries $num_text_queries \
         --output_dir $output_dir --max_gen_tokens $max_gen_tokens \
         --resume $resume --save-freq 5

CUDA_VISIBLE_DEVICES=1 \
python eval.py --metadata datasets/videos_val.json \
               --output_dir $output_dir --resume ${output_dir}/checkpoint.pt \
               --video_feature_type $video_feature_type --video_feature_path $video_feature_path \
               --batch_size 32 --workers 4
