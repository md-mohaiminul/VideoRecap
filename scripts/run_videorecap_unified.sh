batch_size=8
epoch=15

video_mapper_type=qformer
text_mapper_type=qformer

# force_len=100000
force_len=20000
unify_type=caption+clip
hier_type=recur

resume=/checkpoint/mohaiminul/VideoCapHier/outputs/caption/qformer_256/checkpoint.pt
output_dir=/checkpoint/mohaiminul/VideoCapHier/outputs/unified/${unify_type}_${hier_type}_ann

#train
# torchrun --nproc_per_node=8 train_unified_2.py \
#          --batch_size=$batch_size --video_mapper_type $video_mapper_type \
#          --text_mapper_type $text_mapper_type --epochs $epoch \
#          --unify_type $unify_type --hier_type $hier_type --force_len $force_len \
#          --resume $resume --output_dir $output_dir --save-freq 1

# source ~/.bashrc
# conda activate vidsum

CUDA_VISIBLE_DEVICES=2 \
python eval_unified.py --resume pretrained_models/videorecap_unified.pt \
        --output_dir outputs/unified \
        --batch_size 32 --workers 10