

vocab_file:
	python3 util/generate_vocab_file.py --input_file /media/D/DLHLP/text-data.txt --mode character --output_file vocab_file.txt

train_all: train_ASR train_LM
train_ASR:
	python3 main.py --config config/libri/asr_dlhlp.yaml # --load ckpt/asr_dlhlp_sd0/latest.pth
train_ASR_CTC:
	python3 main.py --config config/libri/asr_dlhlp_ctc.yaml  # --load ckpt/asr_dlhlp_ctc_sd0/latest.pth
train_LM:
	python3 main.py --config config/libri/lm_dlhlp.yaml --lm  # --load ckpt/lm_dlhlp_sd0/best_ppx.pth

tensorboard:
	tensorboard --logdir log/ --port 1234

CONFIG=decode
test:
	python3 main.py --config config/libri/$(CONFIG)_dlhlp.yaml --test --njobs 8
test_CTC:
	python3 main.py --config config/libri/$(CONFIG)_dlhlp_ctc.yaml --test --njobs 6
eval:
	python3 eval.py --file result/$(CONFIG)_dlhlp_dev_output.csv
eval_CTC:
	python3 eval.py --file result/$(CONFIG)_dlhlp_ctc_dev_output.csv

submit:
	python3 format.py result/$(CONFIG)_dlhlp_test_output.csv result/submit.csv
submit_CTC:
	python3 format.py result/$(CONFIG)_dlhlp_ctc_test_output.csv result/submit_ctc.csv
submit_reproduce:
	python3 format.py result/$(CONFIG)_dlhlp_ctc_test_output.csv $(ANS_CSV_PATH)

REPRODUCE:
	bash reproduce.sh /media/D/DLHLP/test ans.csv

