
hw3_bonus:
	python3 bonus/mixing.py
	python3 src/evaluate.py --use_cuda 0 --model_path bonus/model.pth.tar --data_dir bonus/data/meta --cal_sdr 1 --sample_rate 8000 --batch_size 1  #>> bonus/evaluate.log 
	python3 src/separate.py --use_cuda 0 --model_path bonus/model.pth.tar --mix_json bonus/data/meta/mix.json --out_dir bonus/my_result  


