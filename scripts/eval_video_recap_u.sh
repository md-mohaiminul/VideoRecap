
CUDA_VISIBLE_DEVICES=0 \
python eval_unified.py --resume pretrained_models/videorecap_unified.pt \
        --output_dir outputs/unified \
        --batch_size 32 --workers 10