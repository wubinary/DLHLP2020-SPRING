SAVE_PATH=./save/Jun18235446/ckpts/500
SAVE=./save/Jun18235446/ckpts/
STEP=500
main:
	python3 main.py --do_train -batch_size 128 -F_pretrain_iter 500 -train_iter 2000 \
	-iter_D 10 -iter_F 5 -eval_steps 25 \
	-slf_factor 0.25 -cyc_factor 0.5 -adv_factor 1 \
	-embed_size 256 -head 4 -num_layers 4 \
	-learned_pos_embed True --use_wandb
test:
	nohup python3 main.py --do_test -test_out ./submission.txt \
		-preload_F ${SAVE_PATH}_F.pth -preload_D ${SAVE_PATH}_D.pth > output.txt &
part2:
	python3 main.py --part2 --part2_model_dir ${SAVE} --part2_step ${STEP}