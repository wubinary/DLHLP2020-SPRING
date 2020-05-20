import torch
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from mafan import simplify
import pdb


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = BertConfig.from_pretrained('OUT_DIR/config.json')
model = BertForTokenClassification.from_pretrained('OUT_DIR/my_cws_bert.pt', config=config).to(device)



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
    return ','.join(res) + '\n'


sentences = [
'我一直親自指揮、親自部署，我相信只要我們堅定信心、同舟共濟、科學防治、精準施策，我們一定會戰勝這一次疫情。',
'這個聲明讓我再次想起了安徒生的童話《皇帝的新裝》。',
'希望他們能夠聽一聽這個忠告，不要再信口雌黃地抹黑，居心叵測地挑撥，煞有介事地恫嚇。',
'有關部門當然就是有關的部門了。無關的就不能稱為有關部門。所以我建議你還是要向他們詢問。',
'不要搞奇奇怪怪的建築。',
'現在提請表決。同意的代表請舉手。請放下；不同意的請舉手。沒有；棄權的請舉手。沒有。通過！',
'人均國內生產總值接近八千萬美元。',
'我青年時代就對法國文化抱有濃厚興趣，法國的歷史、哲學、文學、藝術深深吸引著我。讀法國近現代史特別是法國大革命史的書籍，讓我豐富了對人類社會政治演進規律的思考。讀孟德斯鳩、伏爾泰、盧梭、狄德羅、聖西門、傅立葉、薩特等人的著作，讓我加深了對思想進步對人類社會進步作用的認識。讀蒙田、拉封丹、莫里哀、司湯達、巴爾扎克、雨果、大仲馬、喬治·桑、福樓拜、小仲馬、莫泊桑、羅曼·羅蘭等人的著作，讓我增加了對人類生活中悲歡離合的感觸。冉阿讓、卡西莫多、羊脂球等藝術形象至今仍栩栩如生地存在於我的腦海之中。欣賞米勒、馬奈、德加、塞尚、莫內、羅丹等人的藝術作品，以及趙無極中西合璧的畫作，讓我提升了自己的藝術鑑賞能力。還有，讀凡爾納的科幻小說，讓我的頭腦充滿了無盡的想像。',
'輕關易道，通商寬衣。',
'因為我那時候，扛兩百斤麥子，十里山路不換肩的。'
]

out = ''
for sent in sentences:
    out += seg(sent)

with open('segmented.csv', 'w') as f:
    f.write(out) 