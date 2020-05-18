'''
Takes raw text and saves BERT-cased features for that text to disk

Adapted from the BERT readme (and using the corresponding package) at

https://github.com/huggingface/pytorch-pretrained-BERT

###
John Hewitt, johnhew@stanford.edu
Feb 2019

'''
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer, BertConfig
from argparse import ArgumentParser
import h5py
import numpy as np
from tqdm import tqdm

argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('output_path')
argp.add_argument('bert_model', help='base or large')
args = argp.parse_args()

# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
if args.bert_model == 'base':
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased', output_hidden_states=True)
  model = BertModel.from_pretrained('bert-base-cased')
  LAYER_COUNT = 12
  FEATURE_COUNT = 768
elif args.bert_model == 'chinese':
  tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', output_hidden_states=True)
  config = BertConfig.from_pretrained('bert-base-chinese')
  config.output_hidden_states = True
  # import pdb
  # pdb.set_trace()
  model = BertModel.from_pretrained('bert-base-chinese', config=config)
  LAYER_COUNT = 12
  FEATURE_COUNT = 768
elif args.bert_model == 'large':
  tokenizer = BertTokenizer.from_pretrained('bert-large-cased', output_hidden_states=True)
  model = BertModel.from_pretrained('bert-large-cased')
  LAYER_COUNT = 24
  FEATURE_COUNT = 1024
else:
  raise ValueError("BERT model must be base or large")

model.eval()

with h5py.File(args.output_path, 'w') as fout:
  for index, line in tqdm(enumerate(open(args.input_path))):
    line = line.strip() # Remove trailing characters
    line = '[CLS] ' + line + ' [SEP]'
    tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [1 for x in tokenized_text]
  
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
  
    with torch.no_grad():
        encoded_layers = model(tokens_tensor, segments_tensors)
    # import pdb
    # pdb.set_trace()
    encoded_layers = encoded_layers[2][:12]
    dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
    dset[:,:,:] = np.vstack([np.array(x) for x in encoded_layers])
  

