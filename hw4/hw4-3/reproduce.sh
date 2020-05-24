python  ./examples/run_squad.py \
	--model_type bert \
	--do_eval \
	--model_name_or_path $1 \
	--config_name ./config.json \
	--tokenizer_name bert-base-chinese \
	--train_file Data/hw4-3_train.json \
	--predict_file $2  \
	--max_seq_length 384 \
	--per_gpu_eval_batch_size 100 \
	--output_dir ./  \

python process_ans.py ./predictions_.json $3