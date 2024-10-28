from datasets import load_dataset
import statistics
import csv
import ast
from scipy import stats
import pickle
import random
from itertools import combinations
import math

random_range = 10e-4
random.seed(0)
def loadData(filename):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def parse_explanation(filename):
    explanation = loadData(filename)
    all_indices = []
    all_attr_value = []
    for i in explanation:
        indices = []
        attr_value = []
        attr_list = i[0]
        for j in attr_list:
            indices.append(j[1][1])
            attr_value.append(j[1][0])
        all_indices.append(indices)
        all_attr_value.append(attr_value)
    return all_indices, all_attr_value


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

def sign_agreement(word_saliency_list_1: list, word_saliency_list_2: list, attr_val1: list, attr_val2: list, k: int):
    top_k_1 = word_saliency_list_1[:k:]
    top_k_2 = word_saliency_list_2[:k:]
    result = []
    for i in top_k_1:
        result.append(i in top_k_2 and attr_val1[top_k_1.index(i)] * attr_val2[top_k_2.index(i)] > 0)
    return result.count(1) / k

def signed_rank_agreement(word_saliency_list_1: list, word_saliency_list_2: list, attr_val1: list, attr_val2: list, k: int):
    top_k_1 = word_saliency_list_1[:k:]
    top_k_2 = word_saliency_list_2[:k:]
    result = []
    for i in top_k_1:
        result.append(i in top_k_2 and attr_val1[top_k_1.index(i)] * attr_val2[top_k_2.index(i)] > 0 and top_k_1.index(i) == top_k_2.index(i))
    return result.count(1) / k
    
def rank_correlation(word_saliency_list_1: list, word_saliency_list_2: list, attr_val1: list, attr_val2: list):
    d1 = dict(zip(word_saliency_list_1, attr_val1))
    d2 = dict(zip(word_saliency_list_2, attr_val2))
    sorted_d1 = dict(sorted(d1.items()))
    sorted_d2 = dict(sorted(d2.items()))
    rankcorr= stats.spearmanr(list(sorted_d1.values()), list(sorted_d2.values()))
    return rankcorr.statistic
    
def pairwise_rank_agreement(word_saliency_list_1: list, word_saliency_list_2: list):
    comb = combinations(word_saliency_list_1, 2)
    result = []
    for i in list(comb):
        if word_saliency_list_2.index(i[0]) < word_saliency_list_2.index(i[1]):
            result.append(1)
        else:
            result.append(0)
    return result.count(1)/len(result)

def IOU(word_saliency_list_1: list, word_saliency_list_2: list, k: int):
    top_k_1 = word_saliency_list_1[:k:]
    top_k_2 = word_saliency_list_2[:k:]
    inter = set(top_k_1).intersection(set(top_k_2))
    union = set(top_k_1).union(set(top_k_2))
    return len(inter)/len(union)



def main():
    size = 100
    dataset = load_dataset('sst', split='test')
    dataset = dataset.shuffle(seed=8)['sentence']
    PE_word_saliency_list, PE_attribute_val = parse_explanation("llama3_pe_expl.pickle")
    EP_word_saliency_list, EP_attribute_val= parse_explanation("llama3_ep_expl.pickle")
    PE_Occlusion_word_saliency_list, PE_Occlusion_attribute_val = parse_explanation("llama_PE_Occlusion.pickle")
    EP_Occlusion_word_saliency_list, EP_Occlusion_attribute_val = parse_explanation("llama_EP_Occlusion.pickle")
    PE_LIME_word_saliency_list, PE_LIME_attribute_val = parse_explanation("llama_LIME_response_PE_0_100.pickle")
    EP_LIME_word_saliency_list, EP_LIME_attribute_val = parse_explanation("llama_LIME_response_EP_0_100.pickle")
    PE_Natural_saliency_list, _ = parse_explanation("llama_topk_expl_PE.pickle")
    EP_Natural_saliency_list, _ = parse_explanation("llama_topk_expl_EP.pickle")

    # print(len(PE_LIME_word_saliency_list))
    # print((EP_LIME_word_saliency_list[0]))

    
    # # generating_PE_LIME_saliency_list(dataset)
    # # print("PE_attribute_val")
    # # print(PE_attribute_val)
    # # print("******************************")
    # # print("EP_attribute_val")
    # # print(EP_attribute_val)
    # # print("******************************")
    # # print("PE_LIME_word_saliency_list")
    # # print(PE_LIME_word_saliency_list)
    # # print("******************************")
    # # print("PE_LIME_attribute_val")
    # # print(PE_LIME_attribute_val)

    # for i in range(100):
    #     if len(PE_word_saliency_list[i]) != len(EP_word_saliency_list[i]) and len(PE_Occlusion_word_saliency_list[i]) != len(EP_Occlusion_word_saliency_list[i]) and len(PE_word_saliency_list[i]) != len(EP_Occlusion_word_saliency_list[i]):
    #         print(i)

    print("Evaluation between (PE_word_saliency_list, PE_Occlusion_word_saliency_list)")
    PE_feature_agreement = []
    PE_rank_agreement = []
    PE_sign_agreement = []
    PE_signed_rank_agreement = []
    PE_rank_correlation = []
    PE_pairwise_rank_agreement = []
    PE_IOU = []
    EP_feature_agreement = []
    EP_rank_agreement = []
    EP_sign_agreement = []
    EP_signed_rank_agreement = []
    EP_rank_correlation = []
    EP_pairwise_rank_agreement = []
    EP_IOU = []
    for i in range(size):
        l = len(dataset[i].split())
        t = math.floor(l * 0.2)
        if t == 0:
            t = 1
        PE_feature_agreement.append(feature_agreement(PE_word_saliency_list[i], PE_Occlusion_word_saliency_list[i], t))
        PE_rank_agreement.append(rank_agreement(PE_word_saliency_list[i], PE_Occlusion_word_saliency_list[i], t))
        PE_sign_agreement.append(sign_agreement(PE_word_saliency_list[i], PE_Occlusion_word_saliency_list[i], PE_attribute_val[i], PE_Occlusion_attribute_val[i], t))
        PE_signed_rank_agreement.append(signed_rank_agreement(PE_word_saliency_list[i], PE_Occlusion_word_saliency_list[i], PE_attribute_val[i], PE_Occlusion_attribute_val[i], t))
        PE_rank_correlation.append(rank_correlation(PE_word_saliency_list[i], PE_Occlusion_word_saliency_list[i], PE_attribute_val[i], PE_Occlusion_attribute_val[i]))
        PE_pairwise_rank_agreement.append(pairwise_rank_agreement(PE_word_saliency_list[i], PE_Occlusion_word_saliency_list[i]))
        PE_IOU.append(IOU(PE_word_saliency_list[i], PE_Occlusion_word_saliency_list[i], t))


        EP_feature_agreement.append(feature_agreement(EP_word_saliency_list[i], EP_Occlusion_word_saliency_list[i], t))
        EP_rank_agreement.append(rank_agreement(EP_word_saliency_list[i], EP_Occlusion_word_saliency_list[i], t))
        EP_sign_agreement.append(sign_agreement(EP_word_saliency_list[i], EP_Occlusion_word_saliency_list[i], EP_attribute_val[i], EP_Occlusion_attribute_val[i], t))
        EP_signed_rank_agreement.append(signed_rank_agreement(EP_word_saliency_list[i], EP_Occlusion_word_saliency_list[i], EP_attribute_val[i], EP_Occlusion_attribute_val[i], t))
        EP_rank_correlation.append(rank_correlation(EP_word_saliency_list[i], EP_Occlusion_word_saliency_list[i], EP_attribute_val[i], EP_Occlusion_attribute_val[i]))
        EP_pairwise_rank_agreement.append(pairwise_rank_agreement(EP_word_saliency_list[i], EP_Occlusion_word_saliency_list[i]))
        EP_IOU.append(IOU(EP_word_saliency_list[i], EP_Occlusion_word_saliency_list[i], t))

    print("PE_feature_agreement",sum(PE_feature_agreement)/len(PE_feature_agreement))
    print("PE_rank_agreement",sum(PE_rank_agreement)/len(PE_rank_agreement))
    print("PE_sign_agreement",sum(PE_sign_agreement)/len(PE_sign_agreement))
    print("PE_signed_rank_agreement",sum(PE_signed_rank_agreement)/len(PE_signed_rank_agreement))
    print("PE_rank_correlation",sum(PE_rank_correlation)/len(PE_rank_correlation))
    print("PE_pairwise_rank_agreement",sum(PE_pairwise_rank_agreement)/len(PE_pairwise_rank_agreement))
    print("PE_IOU",sum(PE_IOU)/len(PE_IOU))
    print("******************************************************************")
    print("EP_feature_agreementeature_agreement",sum(EP_feature_agreement)/len(EP_feature_agreement))
    print("EP_rank_agreement",sum(EP_rank_agreement)/len(EP_rank_agreement))
    print("EP_sign_agreement",sum(EP_sign_agreement)/len(EP_sign_agreement))
    print("EP_signed_rank_agreement",sum(EP_signed_rank_agreement)/len(EP_signed_rank_agreement))
    print("EP_rank_correlation",sum(EP_rank_correlation)/len(EP_rank_correlation))
    print("EP_pairwise_rank_agreement",sum(EP_pairwise_rank_agreement)/len(EP_pairwise_rank_agreement))
    print("EP_IOU",sum(EP_IOU)/len(EP_IOU))

    print("\n\n")
    print("Evaluation between (PE_word_saliency_list, PE_LIME_word_saliency_list), (EP_word_saliency_list, EP_LIME_word_saliency_list)")
    PE_feature_agreement = []
    PE_rank_agreement = []
    PE_sign_agreement = []
    PE_signed_rank_agreement = []
    PE_rank_correlation = []
    PE_pairwise_rank_agreement = []
    PE_IOU = []
    EP_feature_agreement = []
    EP_rank_agreement = []
    EP_sign_agreement = []
    EP_signed_rank_agreement = []
    EP_rank_correlation = []
    EP_pairwise_rank_agreement = []
    EP_IOU = []
    for i in range(size):
        # print(i)
        # print(EP_word_saliency_list[i], EP_LIME_word_saliency_list[i], EP_attribute_val[i], EP_LIME_attribute_val[i])
        l = len(dataset[i].split())
        t = math.floor(l * 0.2)
        if t == 0:
            t = 1
        PE_feature_agreement.append(feature_agreement(PE_word_saliency_list[i], PE_LIME_word_saliency_list[i], t))
        PE_rank_agreement.append(rank_agreement(PE_word_saliency_list[i], PE_LIME_word_saliency_list[i], t))
        PE_sign_agreement.append(sign_agreement(PE_word_saliency_list[i], PE_LIME_word_saliency_list[i], PE_attribute_val[i], PE_LIME_attribute_val[i], t))
        PE_signed_rank_agreement.append(signed_rank_agreement(PE_word_saliency_list[i], PE_LIME_word_saliency_list[i], PE_attribute_val[i], PE_LIME_attribute_val[i], t))
        PE_rank_correlation.append(rank_correlation(PE_word_saliency_list[i], PE_LIME_word_saliency_list[i], PE_attribute_val[i], PE_LIME_attribute_val[i]))
        PE_pairwise_rank_agreement.append(pairwise_rank_agreement(PE_word_saliency_list[i], PE_LIME_word_saliency_list[i]))
        PE_IOU.append(IOU(PE_word_saliency_list[i], PE_LIME_word_saliency_list[i], t))


        EP_feature_agreement.append(feature_agreement(EP_word_saliency_list[i], EP_LIME_word_saliency_list[i], t))
        EP_rank_agreement.append(rank_agreement(EP_word_saliency_list[i], EP_LIME_word_saliency_list[i], t))
        EP_sign_agreement.append(sign_agreement(EP_word_saliency_list[i], EP_LIME_word_saliency_list[i], EP_attribute_val[i], EP_LIME_attribute_val[i], t))
        EP_signed_rank_agreement.append(signed_rank_agreement(EP_word_saliency_list[i], EP_LIME_word_saliency_list[i], EP_attribute_val[i], EP_LIME_attribute_val[i], t))
        EP_rank_correlation.append(rank_correlation(EP_word_saliency_list[i], EP_LIME_word_saliency_list[i], EP_attribute_val[i], EP_LIME_attribute_val[i]))
        EP_pairwise_rank_agreement.append(pairwise_rank_agreement(EP_word_saliency_list[i], EP_LIME_word_saliency_list[i]))
        EP_IOU.append(IOU(EP_word_saliency_list[i], EP_LIME_word_saliency_list[i], t))

    print("PE_feature_agreement",sum(PE_feature_agreement)/len(PE_feature_agreement))
    print("PE_rank_agreement",sum(PE_rank_agreement)/len(PE_rank_agreement))
    print("PE_sign_agreement",sum(PE_sign_agreement)/len(PE_sign_agreement))
    print("PE_signed_rank_agreement",sum(PE_signed_rank_agreement)/len(PE_signed_rank_agreement))
    print("PE_rank_correlation",sum(PE_rank_correlation)/len(PE_rank_correlation))
    print("PE_pairwise_rank_agreement",sum(PE_pairwise_rank_agreement)/len(PE_pairwise_rank_agreement))
    print("PE_IOU",sum(PE_IOU)/len(PE_IOU))
    print("******************************************************************")
    print("EP_feature_agreementeature_agreement",sum(EP_feature_agreement)/len(EP_feature_agreement))
    print("EP_rank_agreement",sum(EP_rank_agreement)/len(EP_rank_agreement))
    print("EP_sign_agreement",sum(EP_sign_agreement)/len(EP_sign_agreement))
    print("EP_signed_rank_agreement",sum(EP_signed_rank_agreement)/len(EP_signed_rank_agreement))
    print("EP_rank_correlation",sum(EP_rank_correlation)/len(EP_rank_correlation))
    print("EP_pairwise_rank_agreement",sum(EP_pairwise_rank_agreement)/len(EP_pairwise_rank_agreement))
    print("EP_IOU",sum(EP_IOU)/len(EP_IOU))


    print("\n\n")
    print("Evaluation between (PE_Occlusion_word_saliency_list, PE_LIME_word_saliency_list), (EP_Occlusion_word_saliency_list, EP_LIME_word_saliency_list)")
    PE_feature_agreement = []
    PE_rank_agreement = []
    PE_sign_agreement = []
    PE_signed_rank_agreement = []
    PE_rank_correlation = []
    PE_pairwise_rank_agreement = []
    PE_IOU = []
    EP_feature_agreement = []
    EP_rank_agreement = []
    EP_sign_agreement = []
    EP_signed_rank_agreement = []
    EP_rank_correlation = []
    EP_pairwise_rank_agreement = []
    EP_IOU = []
    for i in range(size):
        l = len(dataset[i].split())
        t = math.floor(l * 0.2)
        if t == 0:
            t = 1
        PE_feature_agreement.append(feature_agreement(PE_Occlusion_word_saliency_list[i], PE_LIME_word_saliency_list[i], t))
        PE_rank_agreement.append(rank_agreement(PE_Occlusion_word_saliency_list[i], PE_LIME_word_saliency_list[i], t))
        PE_sign_agreement.append(sign_agreement(PE_Occlusion_word_saliency_list[i], PE_LIME_word_saliency_list[i], PE_Occlusion_attribute_val[i], PE_LIME_attribute_val[i], t))
        PE_signed_rank_agreement.append(signed_rank_agreement(PE_Occlusion_word_saliency_list[i], PE_LIME_word_saliency_list[i], PE_Occlusion_attribute_val[i], PE_LIME_attribute_val[i], t))
        PE_rank_correlation.append(rank_correlation(PE_Occlusion_word_saliency_list[i], PE_LIME_word_saliency_list[i], PE_Occlusion_attribute_val[i], PE_LIME_attribute_val[i]))
        PE_pairwise_rank_agreement.append(pairwise_rank_agreement(PE_Occlusion_word_saliency_list[i], PE_LIME_word_saliency_list[i]))
        PE_IOU.append(IOU(PE_Occlusion_word_saliency_list[i], PE_LIME_word_saliency_list[i], t))


        EP_feature_agreement.append(feature_agreement(EP_Occlusion_word_saliency_list[i], EP_LIME_word_saliency_list[i], t))
        EP_rank_agreement.append(rank_agreement(EP_Occlusion_word_saliency_list[i], EP_LIME_word_saliency_list[i], t))
        EP_sign_agreement.append(sign_agreement(EP_Occlusion_word_saliency_list[i], EP_LIME_word_saliency_list[i], EP_Occlusion_attribute_val[i], EP_LIME_attribute_val[i], t))
        EP_signed_rank_agreement.append(signed_rank_agreement(EP_Occlusion_word_saliency_list[i], EP_LIME_word_saliency_list[i], EP_Occlusion_attribute_val[i], EP_LIME_attribute_val[i], t))
        EP_rank_correlation.append(rank_correlation(EP_Occlusion_word_saliency_list[i], EP_LIME_word_saliency_list[i], EP_Occlusion_attribute_val[i], EP_LIME_attribute_val[i]))
        EP_pairwise_rank_agreement.append(pairwise_rank_agreement(EP_Occlusion_word_saliency_list[i], EP_LIME_word_saliency_list[i]))
        EP_IOU.append(IOU(EP_Occlusion_word_saliency_list[i], EP_LIME_word_saliency_list[i], t))

    print("PE_feature_agreement",sum(PE_feature_agreement)/len(PE_feature_agreement))
    print("PE_rank_agreement",sum(PE_rank_agreement)/len(PE_rank_agreement))
    print("PE_sign_agreement",sum(PE_sign_agreement)/len(PE_sign_agreement))
    print("PE_signed_rank_agreement",sum(PE_signed_rank_agreement)/len(PE_signed_rank_agreement))
    print("PE_rank_correlation",sum(PE_rank_correlation)/len(PE_rank_correlation))
    print("PE_pairwise_rank_agreement",sum(PE_pairwise_rank_agreement)/len(PE_pairwise_rank_agreement))
    print("PE_IOU",sum(PE_IOU)/len(PE_IOU))
    print("******************************************************************")
    print("EP_feature_agreementeature_agreement",sum(EP_feature_agreement)/len(EP_feature_agreement))
    print("EP_rank_agreement",sum(EP_rank_agreement)/len(EP_rank_agreement))
    print("EP_sign_agreement",sum(EP_sign_agreement)/len(EP_sign_agreement))
    print("EP_signed_rank_agreement",sum(EP_signed_rank_agreement)/len(EP_signed_rank_agreement))
    print("EP_rank_correlation",sum(EP_rank_correlation)/len(EP_rank_correlation))
    print("EP_pairwise_rank_agreement",sum(EP_pairwise_rank_agreement)/len(EP_pairwise_rank_agreement))
    print("EP_IOU",sum(EP_IOU)/len(EP_IOU))


    print("\n\n")
    print("Evaluation between all PE_Natural_saliency_list & EP_Natural_saliency_list")
    PE_feature_agreement1 = []
    PE_feature_agreement2 = []
    PE_feature_agreement3 = []
    PE_IOU1 = []
    PE_IOU2 = []
    PE_IOU3 = []
    
    EP_feature_agreement1 = []
    EP_feature_agreement2 = []
    EP_feature_agreement3 = []
    EP_IOU1 = []
    EP_IOU2 = []
    EP_IOU3 = []


    for i in range(size):
        l = len(dataset[i].split())
        t = math.floor(l * 0.2)
        if t == 0:
            t = 1
        PE_feature_agreement1.append(feature_agreement(PE_word_saliency_list[i], PE_Natural_saliency_list[i], t))
        PE_IOU1.append(IOU(PE_word_saliency_list[i], PE_Natural_saliency_list[i], t))
        PE_feature_agreement2.append(feature_agreement(PE_Occlusion_word_saliency_list[i], PE_Natural_saliency_list[i], t))
        PE_IOU2.append(IOU(PE_Occlusion_word_saliency_list[i], PE_Natural_saliency_list[i], t))
        PE_feature_agreement3.append(feature_agreement(PE_LIME_word_saliency_list[i], PE_Natural_saliency_list[i], t))
        PE_IOU3.append(IOU(PE_LIME_word_saliency_list[i], PE_Natural_saliency_list[i], t))

        EP_feature_agreement1.append(feature_agreement(EP_word_saliency_list[i], EP_Natural_saliency_list[i], t))
        EP_IOU1.append(IOU(EP_word_saliency_list[i], EP_Natural_saliency_list[i], t))
        EP_feature_agreement2.append(feature_agreement(EP_Occlusion_word_saliency_list[i], EP_Natural_saliency_list[i], t))
        EP_IOU2.append(IOU(EP_Occlusion_word_saliency_list[i], EP_Natural_saliency_list[i], t))
        EP_feature_agreement3.append(feature_agreement(EP_LIME_word_saliency_list[i], EP_Natural_saliency_list[i], t))
        EP_IOU3.append(IOU(EP_LIME_word_saliency_list[i], EP_Natural_saliency_list[i], t))


    print("PE_feature_agreement1",sum(PE_feature_agreement1)/len(PE_feature_agreement1))
    print("PE_feature_agreement2",sum(PE_feature_agreement2)/len(PE_feature_agreement2))
    print("PE_feature_agreement3",sum(PE_feature_agreement3)/len(PE_feature_agreement3))
    print("PE_IOU1",sum(PE_IOU1)/len(PE_IOU1))
    print("PE_IOU2",sum(PE_IOU2)/len(PE_IOU2))
    print("PE_IOU3",sum(PE_IOU3)/len(PE_IOU3))

    print("EP_feature_agreement1",sum(EP_feature_agreement1)/len(EP_feature_agreement1))
    print("EP_feature_agreement2",sum(EP_feature_agreement2)/len(EP_feature_agreement2))
    print("EP_feature_agreement3",sum(EP_feature_agreement3)/len(EP_feature_agreement3))
    print("EP_IOU1",sum(EP_IOU1)/len(EP_IOU1))
    print("EP_IOU2",sum(EP_IOU2)/len(EP_IOU2))
    print("EP_IOU3",sum(EP_IOU3)/len(EP_IOU3))



if __name__ == "__main__":
    main()