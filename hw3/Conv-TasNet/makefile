
hw3_bonus:
	python3 bonus/mixing.py
	python3 src/evaluate.py --use_cuda 1 --model_path bonus/model.pth.tar --data_dir bonus/my_mix/ --cal_sdr 1 --sample_rate 8000 --batch_size 2  #>> bonus/evaluate.log 
	python3 src/separate.py --use_cuda 1 --model_path bonus/model.pth.tar --mix_json bonus/my_mix/mix.json --out_dir bonus/my_result  


