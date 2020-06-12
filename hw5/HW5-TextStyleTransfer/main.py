import os
import torch
import time 
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval
import argparse
import wandb
from test import dev_eval

def main(args):
    # logging
    if args.use_wandb:
        wandb.init(project="style-transfer", config=args)
        #wandb.config.update(vars(args))
        args = wandb.config
        print(args)
    
    train_iters, dev_iters, test_iters, vocab = load_dataset(args)
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(args, vocab).to(args.device)
    model_D = Discriminator(args, vocab).to(args.device)
    print(args.discriminator_method)

    if os.path.isfile(args.preload_F):
        temp = torch.load(args.preload_F)
        model_F.load_state_dict(temp)
    if os.path.isfile(args.preload_D):
        temp = torch.load(args.preload_D)
        model_D.load_state_dict(temp)
    
    if args.do_train:
        train(args, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    if args.do_test:
        dev_eval(args, vocab, model_F, test_iters, 0.5)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument("-data_path", default="./data/yelp/", help="the path of the dataset to train.")
    # parser.add_argument("-log_dir", default="runs/exp", help="")
    parser.add_argument("-save_path", default="./save", help="the path to save the checkpoints.")
    parser.add_argument("-preload_F", default="", help="the path to load pretrained model F.")
    parser.add_argument("-preload_D", default="", help="the path to load pretrained discriminator D.")
    parser.add_argument("-pretrained_embed_path", default="./embedding/", help="the path to pretrained Glove vectors.")
    
    # data
    parser.add_argument("-min_freq", default=3, type=int, help="the minimun frequency for building vocabulary.")
    parser.add_argument("-max_length", default=16, type=int, help="the maximum sentence length.")
    
    # model
    parser.add_argument("-discriminator_method", help="the type of discriminator ('Multi' or 'Cond')", default="Multi")
    parser.add_argument("--load_pretrained_embed", help="whether to load pretrained embeddings.", action="store_true")
    parser.add_argument("-embed_size", help="the dimension of the token embedding", default=256, type=int)
    parser.add_argument("-d_model", help="the dimension of Transformer d_model parameter", default=256, type=int)
    parser.add_argument("-head", help="the number of Transformer attention heads", dest="h", default=4, type=int)
    parser.add_argument("-num_styles", help="the number of styles for discriminator", default=2, type=int)
    parser.add_argument("-num_layers", help="the number of Transformer layers", default=4, type=int)
    
    # training
    parser.add_argument("-batch_size", help="the training batch size", default=64 , type=int)
    parser.add_argument("-lr_F", help="the learning rate for the Style Transformer", default=1e-4 , type=float)
    parser.add_argument("-lr_D", help="the learning rate for the discriminator", default=1e-4 , type=float)
    parser.add_argument("-L2", help="the L2 norm regularization factor", default=0.0 , type=float)
    parser.add_argument("-iter_D", help="the number of the discriminator update steps per training iteration", default=10 , type=int)
    parser.add_argument("-iter_F", help="the number of the Style Transformer update steps per training iteration", default=5 , type=int)
    parser.add_argument("-F_pretrain_iter", help="the number of the Style Transformer pretraining steps (train on self rec loss)", default=500 , type=int)
    parser.add_argument("-train_iter", help="total training iterations", default=2000 , type=int)
    #parser.add_argument("-log_steps", dest="log_steps", default=5 , type=int)
    parser.add_argument("-eval_steps", help="the number of steps to per evaluation", default=25 , type=int)
    parser.add_argument("-learned_pos_embed", help="whether to learn positional embedding", default="1", choices=['1', '0', 'True', 'False'])
    parser.add_argument("-dropout", help="the dropout factor for the whole model", default=0.1, type=float)
    
    parser.add_argument("-slf_factor", help="the weight factor for the self reconstruction loss", default=0.25, type=float)
    parser.add_argument("-cyc_factor", help="the weight factor for the cycle reconstruction loss", default=0.5, type=float)
    parser.add_argument("-adv_factor", help="the weight factor for the style controlling loss", default=1, type=float)
    
    # parser.add_argument("-inp_shuffle_len", dest="inp_shuffle_len", default=0, type=int)
    # parser.add_argument("-inp_unk_drop_fac", dest="inp_unk_drop_fac", default=0, type=float)
    # parser.add_argument("-inp_rand_drop_fac", dest="inp_rand_drop_fac", default=0, type=float)
    parser.add_argument("-inp_drop_prob", help="the initial word dropout rate,"
        " which will gradually increase to 2x over the course of training", default=0.1, type=float)
    parser.add_argument("-temp", help="the initial softmax temperature,"
        " which will gradually decrease to 0.5x over the course of training", default=1.0, type=float)
    
    # others
    parser.add_argument("--do_train", help="run training algorithm.", action="store_true")
    parser.add_argument("--do_test", help="run inference on test set.", action="store_true")
    parser.add_argument("-test_out", help="output path for inference.", default="./submission.txt")
    parser.add_argument("--use_wandb", help="log training with wandb, "
        "requires wandb, install with \"pip install wandb\"", action="store_true")
    
    parser.add_argument("--use_gumbel", help="handle discrete part in another way", action="store_true")
    parser.add_argument("--part2", help="run part2", action="store_true")
    parser.add_argument("--part2_model_dir", help="the models use in part2", type=str)
    parser.add_argument("--part2_step", help="the trained step", type=int)


    args = parser.parse_args()
    args.drop_rate_config = [(1, 0), (2, args.train_iter)] # (rate, step), ...
    args.temperature_config = [(args.temp, 0), (0.5*args.temp, args.train_iter)] # (temp, step) ...
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_classes = args.num_styles + 1 if args.discriminator_method == 'Multi' else 2
    args.learned_pos_embed = args.learned_pos_embed.lower() in ("true", "1")
    
    if args.part2:
        from part2 import part2
        assert args.part2_model_dir, "--part2_model_dir=<trained model dir> is needed" 
        # example: 'save/Feb15203331/ckpts/'
        assert args.part2_step, "--part2_step=<trained step> is needed" 
        # example: 1300
        part2(args)
    else:
        main(args)
