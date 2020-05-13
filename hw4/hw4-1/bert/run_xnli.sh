CUDA_VISIBLE_DEVICES=0 python3 run_xnli.py --data_dir /media/D/DLHLP/hw4 \
    --model_type bert \
    --model_name_or_path bert-base-chinese \
    --output_dir /media/D/DLHLP/hw4/hw4-1/output/ \
    --cache_dir /media/D/DLHLP/hw4/hw4-1/dataset_cache \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --warmup_steps 500 \
    --fp16 \
    --language zh \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 48 \
    --per_gpu_eval_batch_size 48 

