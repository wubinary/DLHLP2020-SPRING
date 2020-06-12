from train import *

def dev_eval(config, vocab, model_F, test_iters, temperature):
    model_F.eval()
    vocab_size = len(vocab)
    eos_idx = vocab.stoi['<eos>']

    def inference(data_iter, raw_style):
        gold_text = []
        raw_output = []
        rev_output = []
        for batch in data_iter:
            inp_tokens = batch.text
            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles
        
            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
            
            with torch.no_grad():
                rev_log_probs = model_F(
                    inp_tokens, 
                    None,
                    inp_lengths,
                    rev_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
                
            gold_text += tensor2text(vocab, inp_tokens.cpu())
            raw_output += tensor2text(vocab, raw_log_probs.argmax(-1).cpu())
            rev_output += tensor2text(vocab, rev_log_probs.argmax(-1).cpu())

        return gold_text, raw_output, rev_output

    pos_iter = test_iters.pos_iter
    neg_iter = test_iters.neg_iter
    
    gold_text, raw_output, rev_output = zip(inference(neg_iter, 0), inference(pos_iter, 1))

    with open(config.test_out, 'w') as fw:        

        for idx in range(len(rev_output[0])):
            print('*' * 20, 'neg sample', '*' * 20)
            print('[gold]', gold_text[0][idx])
            print('[raw ]', raw_output[0][idx])
            print('[rev ]', rev_output[0][idx])

            print(rev_output[0][idx], file=fw)

        print('*' * 20, '********', '*' * 20)

        for idx in range(len(rev_output[1])):
            print('*' * 20, 'pos sample', '*' * 20)
            print('[gold]', gold_text[1][idx])
            print('[raw ]', raw_output[1][idx])
            print('[rev ]', rev_output[1][idx])

            print(rev_output[1][idx], file=fw)

        print('*' * 20, '********', '*' * 20)