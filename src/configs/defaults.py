import argparse

def defaultConfigs():
    parser = argparse.ArgumentParser(description='Hierarchical Video Captioning', add_help=False)
    # Data
    parser.add_argument('--dataset', default='segment_description', type=str, 
                        choices=['clip_caption', 'segment_description', 'unified', 'video_summary', 'msr_vtt', 'youcook2', 'es_clip'])
    parser.add_argument('--metadata', default='datasets/summaries_train.pkl', type=str, help='path to metadata file')
    parser.add_argument('--metadata_pseudo', default=None, type=str, help='path to metadata file')
    parser.add_argument('--pseudo_data_sampling', default='random', type=str, choices=['random', 'standard'])
    parser.add_argument('--video_feature_type', default=None, type=str, choices=['pixel', 'cls', None])
    parser.add_argument('--num_video_feat', default=40, type=int, 
                        help='number of frames for clips, number of features for segments or videos')
    parser.add_argument('--chunk_len', default=300, type=int)
    parser.add_argument('--video_loader_type', default='decord', choices=['decord', 'moviepy'], type=str)
    parser.add_argument('--video_feature_width', default=768, type=int)
    parser.add_argument('--video_feature_path', default='/checkpoint/mohaiminul/VideoCapHier/datasets/features/cls', type=str)
    parser.add_argument('--video_sampling_type', default='masking', type=str, choices=['mean', 'uniform', 'random', 'masking'])
    parser.add_argument('--text_feature_type', default=None, type=str)
    parser.add_argument('--num_text_feat', default=1024, type=int)
    parser.add_argument('--text_feature_path', default=None, type=str)
    parser.add_argument('--text_feature_width', default=768, type=int)
    parser.add_argument('--part', default=None, type=int)
    
    parser.add_argument('--unify_type', default=None, type=str)
    parser.add_argument('--force_len', default=None, type=int)
    parser.add_argument('--hier_type', default='recur', type=str, choices=['recur', 'non_recur'])
    
    # Model
    # parser.add_argument('--use_vision_model', action='store_true', help='Use vision encoder')
    parser.add_argument('--vision_model_type', default='timesformer', type=str, choices=['timesformer', 'clip_b16'])
    parser.add_argument('--decoder_name', default='gpt2', type=str)
    parser.add_argument('--cross_attn_freq', default=1, type=int)
    parser.add_argument('--text_width', default=768, type=int)
    parser.add_argument('--query_width', default=768, type=int)
    parser.add_argument('--max_gen_tokens', default=77, type=int)
    parser.add_argument('--num_video_queries', default=256, type=int)
    parser.add_argument('--video_mapper_type', default=None, type=str, choices=['perciever','qformer', 'transformer', None])
    parser.add_argument('--num_text_queries', default=256, type=int)
    parser.add_argument('--text_mapper_type', default=None, type=str, choices=['perciever','qformer', None])
    parser.add_argument('--share_mapper', action='store_true', help='Use same mapper for video and text')
    parser.add_argument('--resume', default=None, type=str, help='path to resume from')
    #             #pretrained_models/vclm_openai_timesformer_base_gpt2_base.pt_ego4d.jobid_319630.ep_0002.md5sum_68a71f.pth
    #             #outputs_summary/distil/distil_bert_pseudo/checkpoint.pt
    parser.add_argument('--video_encoder_ckpt', default='/checkpoint/mohaiminul/VideoCapHier/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth', type=str, help='Contrastive VLP pretrained weights')
                
    
    parser.add_argument('--drop-path-rate', default=0., type=float, help='DropPath rate')
    parser.add_argument('--find-unused-parameters', action='store_true', help='do this during DDP (useful for models with tied weights)')
    parser.add_argument('--finetune_mapper', action='store_true', help='finetune the self-attention layers on distil-bert mapper')
    parser.add_argument('--use_lora', action='store_true', help='Freeze entire text decoder')
    parser.add_argument('--freeze_lm_entire', action='store_true', help='Freeze entire text decoder')
    
    
    # Training
    parser.add_argument('--output_dir', default='outputs/test')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='number of samples per-device/per-gpu')
    
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--fix-lr', action='store_true', help='disable cosine lr decay if set True')
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--lr_scheduler_type', default='cosine', type=str, choices=['cosine', 'linear', None])
    parser.add_argument('--lr_decay', default=0.9, type=float, help='the decay rate of learning rate per epoch')
    parser.add_argument('--lr_step_size', default=1, type=int, help='period of learning rate decay')
    
    parser.add_argument('--clip-grad-type', default='norm', choices=['norm', 'value'])
    parser.add_argument('--clip-grad-value', default=1.0, type=float, help='')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--eval-in-middle-freq', default=-1, type=int)
    parser.add_argument('--save-freq', default=5, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--use-zero', action='store_true', help='use ZeroRedundancyOptimizer to save memory')
    parser.add_argument('--use-checkpoint', action='store_true', help='use gradient checkpointing during training for significantly less GPU usage')
    parser.add_argument('--use-half', action='store_true', help='evaluate using half-precision')
    
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--workers', default=10, type=int, metavar='N', help='number of data loading workers per process')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    
    return parser