from datasets import load_dataset
import statistics
from messages import *
from explanation import *
import csv
import ast
from scipy import stats
import pickle
import random
from itertools import combinations


random_range = 10e-4
random.seed(0)

def generating_PE_word_saliency_list():
    filename="gpt_response_PE.pickle"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)
    new_expl = []
    for sentence, response in expl.items():
        # print(response)
        # print(analyze_pe_result(sentence.split(), response))
        new_expl.append(list(analyze_pe_result(sentence.split(), response)[1].keys()))
    return(new_expl)

def generating_EP_word_saliency_list():
    filename="gpt_response_EP.pickle"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)
    
    new_expl = []
    for sentence, response in expl.items():
        # print(response)
        # print(analyze_ep_result(sentence.split(), response))
        new_expl.append(list(analyze_ep_result(sentence.split(), response)[1].keys()))
    return(new_expl)

def generating_PE_Occlusion_saliency_list():
    filename = "PE_Occlusion_Result_Predict"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)

    new_expl = []
    for response in expl[1]:
        for k, v in response.items():
            response[k] = v + random.uniform(-1 * random_range, random_range)
        new_expl.append(list(dict(sorted(response.items(), key=lambda item: item[1], reverse=True)).keys()))
    return new_expl

def generating_EP_Occlusion_saliency_list():
    filename = "EP_Occlusion_Result_Predict"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)

    new_expl = []
    for response in expl[1]:
        for k, v in response.items():
            response[k] = v + random.uniform(-1 * random_range, random_range)
        new_expl.append(list(dict(sorted(response.items(), key=lambda item: item[1], reverse=True)).keys()))
    return new_expl

def feature_agreement(word_saliency_list_1: list, word_saliency_list_2: list, k: int):
    top_k_1 = word_saliency_list_1[:k:]
    top_k_2 = word_saliency_list_2[:k:]
    inter = set(top_k_1).intersection(set(top_k_2))
    return len(inter) / k

def rank_agreement(word_saliency_list_1: list, word_saliency_list_2: list, k: int):
    top_k_1 = word_saliency_list_1[:k:]
    top_k_2 = word_saliency_list_2[:k:]
    result = []
    for i in top_k_1:
        result.append(i in top_k_2 and top_k_1.index(i) == top_k_2.index(i))
    return result.count(1) / k
    
def rank_correlation(word_saliency_list_1: list, word_saliency_list_2: list):
    rankcorr= stats.spearmanr(word_saliency_list_1, word_saliency_list_2)
    return rankcorr
    
def pairwise_rank_agreement(word_saliency_list_1: list, word_saliency_list_2: list):
    comb = combinations(word_saliency_list_1, 2)
    result = []
    for i in list(comb):
        if word_saliency_list_2.index(i[0]) < word_saliency_list_2.index(i[1]):
            result.append(1)
        else:
            result.append(0)
    return result.count(1)/len(result)


def main():
    size = 100
    dataset = load_dataset('sst', split='test')
    dataset = dataset.shuffle(seed=8)['sentence']
    PE_word_saliency_list = generating_PE_word_saliency_list()
    EP_word_saliency_list = generating_EP_word_saliency_list()
    PE_Occlusion_word_saliency_list = generating_PE_Occlusion_saliency_list()
    EP_Occlusion_word_saliency_list = generating_EP_Occlusion_saliency_list()
    # print("PE_word_saliency_list")
    # print(PE_word_saliency_list)
    # print("******************************")
    # print("EP_word_saliency_list")
    # print(EP_word_saliency_list)
    # print("******************************")
    # print("PE_Occlusion_word_saliency_list")
    # print(PE_Occlusion_word_saliency_list)
    # print("******************************")
    # print("EP_Occlusion_word_saliency_list")
    # print(EP_Occlusion_word_saliency_list)

    # for i in range(100):
    #     if len(PE_word_saliency_list[i]) != len(EP_word_saliency_list[i]) and len(PE_Occlusion_word_saliency_list[i]) != len(EP_Occlusion_word_saliency_list[i]) and len(PE_word_saliency_list[i]) != len(EP_Occlusion_word_saliency_list[i]):
    #         print(i)
        

if __name__ == "__main__":
    main()