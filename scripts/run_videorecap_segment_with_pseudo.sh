# Train videorecap with the manually annotated data + LLM-generated pseudo-annotation

epochs=15
batch_size=32 # 4
dataset=segment_description
metadata=datasets/segments_train.pkl
metadata_pseudo=datasets/segments_train_pseudo.pkl

video_feature_type=cls
num_video_feat=40
num_video_queries=32
video_feature_path=/data/mmiemon/LaVila/datasets/features/vclm_base/cls
video_mapper_type=qformer
video_sampling_type=uniform

text_feature_type=token
num_text_feat=512
num_text_queries=256
text_mapper_type=qformer

output_dir=outputs/videorecap/segments_pseudo
resume=pretrained_models/videorecap/videorecap_clip.pt

torchrun --nproc_per_node=8 train.py \
         --epochs $epochs --batch_size $batch_size \
         --dataset $dataset --metadata $metadata --metadata_pseudo $metadata_pseudo \
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
