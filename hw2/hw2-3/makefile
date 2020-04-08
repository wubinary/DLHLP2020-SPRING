
feature_extraction:
	python3 analyzer.py --dir_to_wav ./dataset/Corpus/wav48 --dir_to_bin ./dataset/Corpus/bin 

collect_stats:
	python3 build.py --train_file_pattern "./dataset/Corpus/bin/Training Set/*/*.bin" --corpus_name dlhlp

train:
	python3 main-vawgan.py --corpus_name dlhlp --architecture architecture-vawgan-dlhlp.json

conversion: conversion_p1 conversion_p2
conversion_p1:
	python3 convert.py --corpus_name dlhlp --src p1 --trg p2 \
	--model_path ./ckpt/model_16.pt \
	--file_pattern "./dataset/Corpus/bin/Testing Set/{}/*.bin"

conversion_p2:
	python3 convert.py --corpus_name dlhlp --src p2 --trg p1 \
	--model_path ./ckpt/model_16.pt \
	--file_pattern "./dataset/Corpus/bin/Testing Set/{}/*.bin"