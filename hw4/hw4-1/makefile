
train:
	CUDA_VISIBLE_DEVICES=0 python3 bert/run_xnli.py --data_dir /media/D/DLHLP/hw4 \
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

generate_similarity_data_pretrained:
	CUDA_VISIBLE_DEVICES=0 python3 ./bert/generate-similarity-data.py \
	--data_dir /media/D/DLHLP/hw4 \
    --model_type bert \
    --model_name_or_path /media/D/DLHLP/hw4/hw4-1/output/checkpoint-400/pytorch_model.bin \
    --language zh \
	--train_language zh \
	--tokenizer_name bert-base-chinese \
	--config_name /media/D/DLHLP/hw4/hw4-1/output/checkpoint-400/config.json \
	--output_dir /media/D/DLHLP/hw4/hw4-1/output2/ \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --warmup_steps 500 \
    --fp16 \
    --overwrite_output_dir \
	--overwrite_cache \
    --per_gpu_train_batch_size 48 \
    --per_gpu_eval_batch_size 48 

generate_similarity_data_fintuned:
	CUDA_VISIBLE_DEVICES=0 python3 ./bert/generate-similarity-data.py \
	--data_dir /media/D/DLHLP/hw4 \
    --model_type bert \
    --model_name_or_path /media/D/DLHLP/hw4/hw4-1/output/checkpoint-24400/pytorch_model.bin \
    --language zh \
	--train_language zh \
	--tokenizer_name bert-base-chinese \
	--config_name /media/D/DLHLP/hw4/hw4-1/output/checkpoint-24400/config.json \
	--output_dir /media/D/DLHLP/hw4/hw4-1/output2/ \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --warmup_steps 500 \
    --fp16 \
    --overwrite_output_dir \
	--overwrite_cache \
    --per_gpu_train_batch_size 48 \
    --per_gpu_eval_batch_size 48 


#    --cache_dir /media/D/DLHLP/hw4/hw4-1/dataset_cache \

calculate_similarity:
	python3 ./bert/similarity_student.py 



