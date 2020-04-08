
train:
	python3 main.py

test: test_p1 test_p2
test_p1:
	python3 main.py --mode test --test_iters 160000 --src_speaker p1 --trg_speaker "['p2']"
test_p2:
	python3 main.py --mode test --test_iters 160000 --src_speaker p2 --trg_speaker "['p1']"