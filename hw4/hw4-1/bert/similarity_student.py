import numpy as np
import pickle as pk
import torch
import random
import pdb
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import IPython
from sklearn.decomposition import PCA, TruncatedSVD

def load_data(path):
    data = pk.load(open(path,"rb"))
    return data

def preprocessing(data,layer_index):
    x_layer = []
    # generate emnbedding data of specify layer on the saved data
    for x in data:
        for word_index in range(x["input_ids"].shape[0]):
            word = x["input_ids"][word_index]
            if word == 101 or word == 0 or word == 102:
                continue
            embedding = x["layer_"+str(layer_index)][word_index]
            x_layer += [(word, embedding)]
    return x_layer

def preprocessing_intra_sentence(data,layer_index):
    x_layer = []
    # generate data of specify layer
    for x in data:
        sentence_embeddings = []
        count = 0
        for word_index in range(x["input_ids"].shape[0]):
            word = x["input_ids"][word_index]
            if word == 101 or word == 0 or word == 102:
                continue
            embedding = x["layer_"+str(layer_index)][word_index]
            sentence_embeddings += [embedding]
            count += 1
        sentence_embedding = np.mean(sentence_embeddings,axis=0)
        for word_index in range(x["input_ids"].shape[0]):
            word = x["input_ids"][word_index]

            #skip [PAD], [SEP], [CLS]
            if word == 101 or word == 0 or word == 102:
                continue
            embedding = x["layer_"+str(layer_index)][word_index]
            x_layer += [(word, embedding, sentence_embedding,count)]
    return x_layer

#Question 2 - main
def Anisotropy_function(version):
    """
    version have three option == "self-sim" , "intra-sentence-sim", "MEV(BONUS)"
    """
    # TA: You may need to modify to your pretrained data path
    samples =load_data("xnli-pretrained-example-data.p")

    #Pretrained version
    record = []

    for i in range(0, 13):
        cos = Anisotropy(samples,i,version)
        record += [(i,cos)]

    plt.plot([x[0] for x in record], [y[1] for y in record], "o-", label="pretrained Model", color="purple")


    #Finetune version
    record = []
    #You may need to modify to your finetune data path
    samples = load_data("xnli-finetune-example-data.p")

    for i in range(0, 13):
        cos = Anisotropy(samples,i,version)
        record += [(i,cos)]

    plt.plot([x[0] for x in record], [y[1] for y in record], "o-", label="Finetuned Model", color="green")
    plt.legend(loc='upper right')
    plt.savefig("picture/"+version+"_Anisotropy.png")
    print("finish anisotropy!")
    plt.clf()
    return

# Question 2
def Anisotropy(data,layer_index,version):
    if version == "self-sim" or version == "intra-sentence-sim":
        x_layer = preprocessing(data,layer_index)
        # calculate anisotropy
        average_cos = []
        for _ in range(1000):
            two_words =random.sample(x_layer,2)
            cos = cosine_similarity_Anisotropy(two_words)
            average_cos += [ cos ]
        average_cos = np.array(average_cos)
        mean = np.mean(average_cos)
    else:
        x_layer = preprocessing(data,layer_index)
        # calculate anisotropy on MEV
        mean = MEV_Anisotropy(x_layer)

    return mean

# Question 2
def cosine_similarity_Anisotropy(two_words):

    """
    Todo: return two word cosine similarity
    """
    #print(two_words[0][1].shape) # (word, embedding)
    cos = cosine_similarity(two_words[0][1].reshape(1,-1),two_words[1][1].reshape(1,-1))
    return cos

# Question 3 -main - IntraSentenceSimilarity
def IntraSentenceSimilarity_function():
    # TA: You may need to modify to your pretrained data path
    samples =load_data("xnli-pretrained-example-data.p")

    #Pretrained version
    record = []
    for i in tqdm(range(0, 13)):
       cos = IntraSentenceSimilarity(samples,i)
       record += [(i,cos)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="pretrained Model", color="purple")

    #finetune version
    record = []
    #You may need to modify to your finetune data path
    samples = load_data("xnli-finetune-example-data.p")
    for i in tqdm(range(0, 13)):
       cos = IntraSentenceSimilarity(samples,i)
       record += [(i,cos)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="finetuned Model", color="green" )
    plt.legend(loc='upper right')
    plt.savefig("picture/Intra-sentence-similarity.png")
    print("finish Intra-sentence-similarity!")
    plt.clf()
    return
# Question 3 - IntraSentenceSimilarity
def IntraSentenceSimilarity(data,layer_index):

    x_layer = preprocessing_intra_sentence(data,layer_index) # (word, embedding, sentence_embedding,count)

    average_cos = []
    for x in x_layer:
        """
        Todo: calculate intra-sentence cosine similarity
        """
        cos = cosine_similarity(x[1].reshape(1,-1),x[2].reshape(1,-1))
        average_cos += [ np.mean(cos)/x[3] ]
    mean = sum(average_cos) / len(data)
    return mean

# Question 3 - main SelfSimilarity
def SelfSimilarity_function():

    # TA: You may need to modify to your pretrained data path
    samples = load_data("xnli-pretrained-example-data.p")

    #Pretrained version
    record = []
    for i in range(0,13):
        self_similarity(samples,i,"pretrained")
        layer_self_similarity = calculate_self_similarity("pretrained",i)
        record += [(i,layer_self_similarity)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="pretrained Model", color="purple" )

    print('finish pretrained SelfSimilarity_function')
    #Finetuned version
    record = []

    #You may need to modify to your finetune data path
    samples = load_data("xnli-finetune-example-data.p")
    for i in range(0, 13):
        self_similarity(samples,i,"finetune")
        layer_self_similarity = calculate_self_similarity("finetune",i)
        record += [(i,layer_self_similarity)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="finetuned Model", color="green" )
    plt.legend(loc='upper right')
    plt.savefig("picture/Self-similarity.png")
    print("finish Self-similarity!")
    plt.clf()
    return


# Question 3 - self_similarity
def calculate_self_similarity(function,layer_index):
    data = pk.load(open(function+"/layer_index_"+str(layer_index)+"_data.p",'rb'))
    total_average_cos = []
    for key in tqdm(data.keys()):
        same_word_embeddings=data[key]
        average_cos = []
        """
        Todo:
        calculate the mean cosine similarity of same word but different context
        Hint: You can write new function to do this or sklearn cosine similarity

        """
        for i in range(len(same_word_embeddings)):
            for j in range(i+1,len(same_word_embeddings)):
                average_cos +=  [ cosine_similarity(same_word_embeddings[i].reshape(1,-1),same_word_embeddings[j].reshape(1,-1)) ]
        average_cos = np.array(average_cos)
        mean = np.mean(average_cos)
        total_average_cos += [ mean ]
    return np.mean(np.array(total_average_cos))

# Question 3
def self_similarity(data,layer_index,function):
    x_layer = preprocessing(data,layer_index) # (word, embedding)

    words = [x[0] for x in x_layer]
    stochastic =Counter(words)
    remove_key = []

    # if word appear less than 2 times in example dataset,
    # we remove it to calculate self-similarity

    for key in stochastic.keys():
        if stochastic[key] < 2:
            remove_key += [key]
    for key in remove_key:
        del stochastic[key]

    select_numbers = stochastic.keys()
    # print(len(select_numbers))

    select_numbers_groups = []
    #build a dictionary:
    init = dict(zip(select_numbers,[[] for x in range(0,len(select_numbers))]))
    # print(init)
    for sample in tqdm(x_layer):
        if sample[0] in init.keys():
            init[sample[0]] += [sample[1]]

    pk.dump(init,open(function+"/layer_index_"+str(layer_index)+"_data.p","wb"))

# Question 4 -main - AnisotropyAdjustedSelfSimilarity
def AnisotropyAdjustedSelfSimilarity_function():
    # TA: You may need to modify to your pretrained data path
    samples =load_data("xnli-pretrained-example-data.p")

    #Pretrained version
    record = []
    for i in range(0, 13):
        cos = Anisotropy(samples,i,version="self-sim")
        self_similarity(samples,i,"pretrained")
        layer_self_similarity = calculate_self_similarity("pretrained",i)
        record += [(i,layer_self_similarity - cos)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="pretrained Model", color="purple")

    #finetune version
    record = []
    #You may need to modify to your finetune data path
    samples = load_data("xnli-finetune-example-data.p")
    for i in range(0, 13):
        cos = Anisotropy(samples,i,version="self-sim")
        self_similarity(samples,i,"finetune")
        layer_self_similarity = calculate_self_similarity("finetune",i)
        record += [(i,layer_self_similarity - cos)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="finetuned Model", color="green")
    plt.legend(loc='upper right')
    plt.savefig("picture/Anisotropy-adjusted-self-similarity.png")
    print("finish Anisotropy-adjusted-self-similarity!")
    plt.clf()
    return

# Question 4 -main - AnisotropyAdjustedIntraSentenceSimilarity
def AnisotropyAdjustedIntraSentenceSimilarity_function():
    # TA: You may need to modify to your pretrained data path
    samples =load_data("xnli-pretrained-example-data.p")

    #Pretrained version
    record = []
    for i in tqdm(range(0, 13)):
       cos = Anisotropy(samples,i,version="intra-sentence-sim")
       IntraSentenceSimilarity_cos = IntraSentenceSimilarity(samples,i)
       record += [(i,IntraSentenceSimilarity_cos - cos)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="pretrained Model", color="purple")

    #Finetuned version
    record = []
    #You may need to modify to your finetune data path
    samples = load_data("xnli-finetune-example-data.p")
    for i in tqdm(range(0, 13)):
       cos = Anisotropy(samples,i,version="intra-sentence-sim")
       IntraSentenceSimilarity_cos = IntraSentenceSimilarity(samples,i)
       record += [(i,IntraSentenceSimilarity_cos - cos)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="finetuned Model", color="green" )
    plt.legend(loc='upper right')
    plt.savefig("picture/Anistropy-adjusted-Intra-sentence-similarity.png")
    print("finish Anistropy-adjusted-Intra-sentence-similarity!")
    plt.clf()
    return




### Bonus:
def MEV(function,layer_index):
    data = pk.load(open(function+"/layer_index_"+str(layer_index)+"_data.p",'rb'))
    total_MEV = []
    for key in tqdm(list(data.keys())[:500]):
        same_word_embeddings=data[key]
        matrix=np.stack(same_word_embeddings,axis=0)
        """
        Todo: do PCA n_components=1, and return each word 's variance_explained_ratio of pca
        """
    return np.mean(np.array(total_MEV))


#Bonus - 1:
def MEV_Anisotropy(data):
    total_MEV = []
    all_data = []
    for i in range(10000):
        all_data += [data[i][1]]

    matrix=np.stack(all_data,axis=0)

    """
    Todo: do Truncated SVD n_components=100, and return average 's variance_explained_ratio of the first component of Truncated SVD
    """

    return total_MEV

### Bonus - 2:
def MaximumExplainableVariance_function():
    samples = load_data("xnli-pretrained-example-data.p")

    #Pretrained version
    record = []
    for i in range(0,13):
        layer_MEV = MEV("pretrained",i)
        record += [(i,layer_MEV)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="pretrained Model" )

    #Finetuned version
    record = []
    #You may need to modify to your finetune data path
    samples = load_data("xnli-finetune-example-data.p")
    for i in range(0, 13):
        layer_MEV = MEV("finetune",i)
        record += [(i,layer_MEV)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="finetuned Model" )
    plt.legend(loc='upper right')
    plt.savefig("picture/Maximum-explainable-variance.png")
    print("finish Maximum-explainable-variance!")
    plt.clf()
    return

### Bonus - 3:
def AnisotropyAdjustedMEV_function():
    samples = load_data("xnli-pretrained-example-data.p")

    #Pretrained version
    record = []
    for i in range(0,13):
        layer_MEV = MEV("pretrained",i)
        mean_MEV = Anisotropy(samples,i,version="MEV")
        record += [(i,layer_MEV- mean_MEV)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="pretrained Model" )

    #finetune version
    record = []
    #You may need to modify to your finetune data path
    samples = load_data("xnli-finetune-example-data.p")
    for i in range(0, 13):
        layer_MEV = MEV("finetune",i)
        mean_MEV = Anisotropy(samples,i,version="MEV")
        record += [(i,layer_MEV-mean_MEV)]
    plt.plot([x[0] for x in record], [y[1] for y in record], "o-",label="finetuned Model" )
    plt.legend(loc='upper right')
    plt.savefig("picture/Adjusted-Maximum-explainable-variance.png")
    print("finish Adjusted Maximum-explainable-variance!")
    plt.clf()
    return

if __name__ == "__main__":

    #Question 2
    #Anisotropy_function(version="self-sim")
    #Anisotropy_function(version="intra-sentence-sim")

    #Question 3
    #SelfSimilarity_function()
    IntraSentenceSimilarity_function()

    #Question 4
    AnisotropyAdjustedIntraSentenceSimilarity_function()
    AnisotropyAdjustedSelfSimilarity_function()

    #Bonus
    #Anisotropy_function(version="MEV")

    #Bonus
    #MaximumExplainableVariance_function()

    #Bonus
    #AnisotropyAdjustedMEV_function()



