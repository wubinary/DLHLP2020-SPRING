python  ./examples/run_squad.py \
	--model_type bert \
	--do_eval \
	--model_name_or_path ./output/MY_MODEL \
	--tokenizer_name bert-base-chinese \
	--train_file Data/hw4-3_train.json \
	--predict_file Data/hw4-3_test.json \
	--max_seq_length 384 \
	--per_gpu_eval_batch_size 100 \
	--output_dir ./output/MY_MODEL \

