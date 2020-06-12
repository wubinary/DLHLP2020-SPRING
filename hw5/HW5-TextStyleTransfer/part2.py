import os
import torch
import time 
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval
import argparse
from train import batch_preprocess, get_lengths
from utils import tensor2text
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def log(f, *args):
    print(*args)
    print(*args, file=f)

def show_attn(src, output, attention, title, output_name=None):

    src = ['[style]'] + src

    assert len(src) == attention[0].shape[1]
    assert len(output) == attention[0].shape[0]


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    axes = [ax1, ax2, ax3, ax4]
    for i in range(len(axes)):
        
        ax = axes[i]
        im = ax.imshow(attention[i], cmap="YlGn")

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("weight", rotation=-90, va="bottom")

        ax.set_xticks(np.arange(len(src)))
        ax.set_yticks(np.arange(len(output)))

        ax.set_xticklabels(src)
        ax.set_yticklabels(output)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(attention.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(attention.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(f'Head {i}')
        fig.tight_layout()
    
    plt.tight_layout()
    if output_name:
        plt.savefig(output_name)

def part2(args):

    ## load model
    #model_prefix = './save/Feb15203331/ckpts/1300'
    model_prefix = os.path.join(args.part2_model_dir, str(args.part2_step))

    args.preload_F = f'{model_prefix}_F.pth'
    args.preload_D = f'{model_prefix}_D.pth'

    ## load data
    train_iters, dev_iters, test_iters, vocab = load_dataset(args)

    ## output dir
    output_dir = 'part2_output'
    os.makedirs(output_dir, exist_ok = True)

    log_f = open(os.path.join(output_dir, 'log.txt'), 'w')

    model_F = StyleTransformer(args, vocab).to(args.device)
    model_D = Discriminator(args, vocab).to(args.device)

    assert os.path.isfile(args.preload_F)
    model_F.load_state_dict(torch.load(args.preload_F))
    assert os.path.isfile(args.preload_D)
    model_D.load_state_dict(torch.load(args.preload_D))
    
    model_F.eval()
    model_D.eval()

    dataset = test_iters
    pos_iter = dataset.pos_iter
    neg_iter = dataset.neg_iter

    pad_idx = vocab.stoi['<pad>'] # 1
    eos_idx = vocab.stoi['<eos>'] # 2
    unk_idx = vocab.stoi['<unk>'] # 0

    
    ## 2-1 attention
    log(log_f, "***** 2-1: Attention *****")

    gold_text = []
    gold_token = []
    rev_output = []
    rev_token = []
    attn_weight = None

    raw_style = 1 ## neg: 0, pos: 1


    for batch in pos_iter:

        inp_tokens = batch.text
        inp_lengths = get_lengths(inp_tokens, eos_idx)
        raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
        rev_styles = 1 - raw_styles

        with torch.no_grad():
            rev_log_probs = model_F(
                inp_tokens, 
                None,
                inp_lengths,
                rev_styles,
                generate=True,
                differentiable_decode=False,
                temperature=1
            )

        rev_attn = model_F.get_decode_src_attn_weight()
        if attn_weight == None:
            attn_weight = rev_attn
        else:
            for layer in range(len(rev_attn)):
                attn_weight[layer] = torch.cat([attn_weight[layer], rev_attn[layer]])


        gold_text += tensor2text(vocab, inp_tokens.cpu())
        rev_idx = rev_log_probs.argmax(-1).cpu()
        rev_output += tensor2text(vocab, rev_idx)

        gold_token.extend([[vocab.itos[j] for j in i] for i in inp_tokens])
        rev_token.extend([[vocab.itos[j] for j in i ] for i in rev_idx])

        break ## select first batch to speed up
    
    # attn_weight[layer] = (Batch, Head, Source, Style+Target)

    idx = np.random.randint(len(rev_output))
    log(log_f, '*' * 20, 'pos sample', '*' * 20)
    log(log_f, '[gold]', gold_text[idx])
    log(log_f, '[rev ]', rev_output[idx])
    for l in range(len(attn_weight)):
        output_name = os.path.join(output_dir, f'problem1_attn_layer{l}.png')
        show_attn(gold_token[idx], rev_token[idx], attn_weight[l][idx], 'attention map', output_name)
        log(log_f, f'save attention figure at {output_name}')
    
    log(log_f, '***** 2-1 end *****')
    log(log_f)


    ## 2-2. tsne
    log(log_f, "***** 2-2: T-sne *****")
    features = []
    labels = []

    for batch in pos_iter:

        inp_tokens = batch.text
        inp_lengths = get_lengths(inp_tokens, eos_idx)

        _, pos_features = model_D(inp_tokens, inp_lengths, return_features=True)
        features.extend(pos_features.detach().cpu().numpy())
        labels.extend([0 for i in range(pos_features.shape[0])])

        raw_style = 1
        raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
        rev_styles = 1 - raw_styles

        with torch.no_grad():
            rev_log_probs = model_F(
                inp_tokens, 
                None,
                inp_lengths,
                rev_styles,
                generate=True,
                differentiable_decode=False,
                temperature=1
            )

        rev_tokens = rev_log_probs.argmax(-1)
        rev_lengths = get_lengths(rev_tokens, eos_idx)
        _, rev_features = model_D(rev_tokens, inp_lengths, return_features=True)
        features.extend(rev_features.detach().cpu().numpy())
        labels.extend([1 for i in range(rev_features.shape[0])])
        

    for batch in neg_iter:

        inp_tokens = batch.text
        inp_lengths = get_lengths(inp_tokens, eos_idx)

        _, neg_features = model_D(inp_tokens, inp_lengths, return_features=True)
        features.extend(neg_features.detach().cpu().numpy())
        labels.extend([2 for i in range(neg_features.shape[0])])

        raw_style = 0
        raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
        rev_styles = 1 - raw_styles

        with torch.no_grad():
            rev_log_probs = model_F(
                inp_tokens, 
                None,
                inp_lengths,
                rev_styles,
                generate=True,
                differentiable_decode=False,
                temperature=1
            )

        rev_tokens = rev_log_probs.argmax(-1)
        rev_lengths = get_lengths(rev_tokens, eos_idx)
        _, rev_features = model_D(rev_tokens, inp_lengths, return_features=True)
        features.extend(rev_features.detach().cpu().numpy())
        labels.extend([3 for i in range(rev_features.shape[0])])

    labels = np.array(labels)
    colors = ['red', 'blue', 'orange', 'green']
    classes = ['POS', 'POS -> NEG', 'NEG', 'NEG -> POS']
    X_emb = TSNE(n_components=2).fit_transform(features)
    
    fig, ax = plt.subplots()
    for i in range(4):
        idxs = labels == i
        ax.scatter(X_emb[idxs, 0], X_emb[idxs, 1], color=colors[i], label=classes[i], alpha=0.8, edgecolors='none')
    ax.legend()
    ax.set_title('t-sne of four distributions')
    output_name = os.path.join(output_dir, 'problem2_tsne.png')
    plt.savefig(output_name)
    log(log_f, f'save T-sne figure at {output_name}')
    log(log_f, "***** 2-2 end *****")
    log(log_f)

    # 2-3. mask input tokens
    log(log_f, '***** 2-3: mask input *****')
    raw_style = 1

    for batch in pos_iter:
        inp_tokens = batch.text
        inp_lengths = get_lengths(inp_tokens, eos_idx)
        break ## only select first batch

    sample_idx = np.random.randint(inp_tokens.shape[0])
    inp_token = inp_tokens[sample_idx]
    inp_length = inp_lengths[sample_idx]
    
    inp_tokens = inp_token.repeat(inp_length-2+1, 1) ## mask until '. <eos>' but contain the origin sentence
    for i in range(inp_tokens.shape[0]-1):
        inp_tokens[i+1][i] = unk_idx

    inp_lengths = torch.full_like(inp_tokens[:, 0], inp_length)
    raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
    rev_styles = 1 - raw_styles

    with torch.no_grad():
        rev_log_probs = model_F(
            inp_tokens, 
            None,
            inp_lengths,
            rev_styles,
            generate=True,
            differentiable_decode=False,
            temperature=1
        )

    gold_text = tensor2text(vocab, inp_tokens.cpu(), remain_unk=True)
    rev_idx = rev_log_probs.argmax(-1).cpu()
    rev_output = tensor2text(vocab, rev_idx, remain_unk=True)

    for i in range(len(gold_text)):
        log(log_f, '-')
        log(log_f, '[ORG]', gold_text[i])
        log(log_f, '[REV]', rev_output[i])

    log(log_f, '***** 2-3 end *****')
    log_f.close()