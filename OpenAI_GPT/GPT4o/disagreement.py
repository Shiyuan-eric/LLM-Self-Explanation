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
    
    explanations = {
        "PE": parse_explanation("gpt4o_pe_expl.pickle"),
        "EP": parse_explanation("gpt4o_ep_expl.pickle"),
        "PE_Occlusion": parse_explanation("gpt4o_pe_occlusion.pickle"),
        "EP_Occlusion": parse_explanation("gpt4o_ep_occlusion.pickle"),
        "PE_LIME": parse_explanation("gpt4o_LIME_response_PE_0_100.pickle"),
        "EP_LIME": parse_explanation("gpt4o_LIME_response_EP_0_100.pickle"),
        "PE_Natural": parse_explanation("gpt4o_topk_expl_PE.pickle"),
        "EP_Natural": parse_explanation("gpt4o_topk_expl_EP.pickle")
    }

    def evaluate_agreements(expl1, expl2, attr1, attr2, size, dataset):
        feature_agreement_scores = []
        rank_agreement_scores = []
        sign_agreement_scores = []
        signed_rank_agreement_scores = []
        rank_correlation_scores = []
        pairwise_rank_agreement_scores = []
        IOU_scores = []

        for i in range(size):
            l = len(dataset[i].split())
            t = max(math.floor(l * 0.2), 1)
            feature_agreement_scores.append(feature_agreement(expl1[i], expl2[i], t))
            rank_agreement_scores.append(rank_agreement(expl1[i], expl2[i], t))
            sign_agreement_scores.append(sign_agreement(expl1[i], expl2[i], attr1[i], attr2[i], t))
            signed_rank_agreement_scores.append(signed_rank_agreement(expl1[i], expl2[i], attr1[i], attr2[i], t))
            rank_correlation_scores.append(rank_correlation(expl1[i], expl2[i], attr1[i], attr2[i]))
            pairwise_rank_agreement_scores.append(pairwise_rank_agreement(expl1[i], expl2[i]))
            IOU_scores.append(IOU(expl1[i], expl2[i], t))

        rank_correlation_scores = [x for x in rank_correlation_scores if not math.isnan(x)]
        return {
            "feature_agreement": sum(feature_agreement_scores) / len(feature_agreement_scores),
            "rank_agreement": sum(rank_agreement_scores) / len(rank_agreement_scores),
            "sign_agreement": sum(sign_agreement_scores) / len(sign_agreement_scores),
            "signed_rank_agreement": sum(signed_rank_agreement_scores) / len(signed_rank_agreement_scores),
            "rank_correlation": sum(rank_correlation_scores) / len(rank_correlation_scores),
            "pairwise_rank_agreement": sum(pairwise_rank_agreement_scores) / len(pairwise_rank_agreement_scores),
            "IOU": sum(IOU_scores) / len(IOU_scores)
        }

    def print_evaluation_results(results, label):
        print(f"Evaluation between {label}")
        for key, value in results.items():
            print(f"{key}: {value}")
        print("******************************************************************")

    evaluations = [
        ("PE vs PE_Occlusion", explanations["PE"], explanations["PE_Occlusion"]),
        ("EP vs EP_Occlusion", explanations["EP"], explanations["EP_Occlusion"]),
        ("PE vs PE_LIME", explanations["PE"], explanations["PE_LIME"]),
        ("EP vs EP_LIME", explanations["EP"], explanations["EP_LIME"]),
        ("PE_Occlusion vs PE_LIME", explanations["PE_Occlusion"], explanations["PE_LIME"]),
        ("EP_Occlusion vs EP_LIME", explanations["EP_Occlusion"], explanations["EP_LIME"]),
        ("PE vs PE_Natural", explanations["PE"], explanations["PE_Natural"]),
        ("EP vs EP_Natural", explanations["EP"], explanations["EP_Natural"]),
        ("PE_Occlusion vs PE_Natural", explanations["PE_Occlusion"], explanations["PE_Natural"]),
        ("EP_Occlusion vs EP_Natural", explanations["EP_Occlusion"], explanations["EP_Natural"]),
        ("PE_LIME vs PE_Natural", explanations["PE_LIME"], explanations["PE_Natural"]),
        ("EP_LIME vs EP_Natural", explanations["EP_LIME"], explanations["EP_Natural"])
    ]

    for label, (expl1, attr1), (expl2, attr2) in evaluations:
        results = evaluate_agreements(expl1, expl2, attr1, attr2, size, dataset)
        print_evaluation_results(results, label)



if __name__ == "__main__":
    main()