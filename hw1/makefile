

vocab_file:
	python3 util/generate_vocab_file.py --input_file /media/D/DLHLP/text-data.txt --mode character --output_file vocab_file.txt 

train_all: train_ASR train_LM 
train_ASR:
	python3 main.py --config config/libri/asr_dlhlp.yaml 
train_LM:
	python3 main.py --config config/libri/lm_dlhlp.yaml --lm 

tensorboard:
	tensorboard --logdir log/ --port 1234 

CONFIG=decode
test:
	python3 main.py --config config/libri/$(CONFIG)_dlhlp.yaml --test --njobs 8
eval:
	python3 eval.py --file result/$(CONFIG)_dlhlp_dev_output.csv 

submit:
	python3 format.py result/$(CONFIG)_dlhlp_test_output.csv result/submit.csv 

