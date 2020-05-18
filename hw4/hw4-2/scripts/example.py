import torch
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from mafan import simplify
import pdb
import sys


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = BertConfig.from_pretrained(sys.argv[1]) #'scripts/config.json')
model = BertForTokenClassification.from_pretrained(sys.argv[2], config=config).to(device) #'scripts/my_cws_bert.pt'


def seg(sentence):
    paraphrase = tokenizer.encode_plus(simplify(sentence), return_tensors="pt")
    paraphrase['attention_mask'][-1] = 0
    # pdb.set_trace()
    for key in paraphrase.keys():
        paraphrase[key] = paraphrase[key].to(device) 

    paraphrase_classification_logits = model(**paraphrase)[0]
    paraphrase_results = paraphrase_classification_logits.argmax(axis=-1)[0]
    paraphrase_results = paraphrase_results[1:-1]
    # pdb.set_trace()

    res = list()
    length = 0
    word = False
    for i in range(len(paraphrase_results)):
        if paraphrase_results[i] == 3:
            res.append(sentence[i])
            length = 0
        if paraphrase_results[i] == 0:
            length += 1
        if paraphrase_results[i] == 1:
            length += 1
        if paraphrase_results[i] == 2:
            res.append(sentence[i-length:i+1])
            length = 0

    print(' '.join(res), '\n')
    return ' '.join(res) + '\n'


sentences = [
'現在防控還不能麻痹，還是不要進行過多的聚集活動。',
'希望同各方一道，繪製「精甚」細膩的工筆畫',
'不要搞奇奇怪怪的建築。',
'沒有可以奉為金科律玉的教科書，也沒有可以對人民頤使氣指的教師爺。',
'天行健，君子以不強自……自強不息。',
'我背過《新華字典》',
'三隻手合力。',
'在人民面前，我們永遠是小學生。',
'別看你今天鬧得歡，小心今後拉清單，這都得應驗的。',
'不要幹這種事情。頭上三尺有神明，一定要有敬畏之心。'
]

out = ''
for sent in sentences:
    out += seg(sent)

with open('data/segmented.txt', 'w') as f:
    f.write(out) 