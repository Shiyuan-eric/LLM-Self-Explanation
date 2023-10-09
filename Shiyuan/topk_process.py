import ast
from tqdm import tqdm
import openai
from retry import retry
import pickle
import random
import time
import pandas as pd
from scipy import stats
import os
from datasets import load_dataset
from explanation import *

openai.api_key = os.getenv("OPENAI_API_KEY")

def generating_natural_PE_saliency_list():
    filename = "parse_topk.pickle"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)
    new_expl = []
    for key, value in expl.items():
        temp = []
        for i in value[0]:
            temp.append(key.split().index(i))
        new_expl.append(temp)
    return new_expl

def generating_natural_EP_saliency_list():
    filename = "parse_topk_EP.pickle"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)
    new_expl = []
    for key, value in expl.items():
        temp = []
        for i in value[0]:
            temp.append(key.split().index(i))
        new_expl.append(temp)
    return new_expl

def topk_comprehensiveness(sentence: str, word_saliency_list: list, PE: bool):
    sentences = sentence.split()
    to_sum = []
    ns = sentences.copy()
    if PE:
        og_score = get_P_E_result_P_Only(sentence=sentence, prompt=p_e_msg(few_shot=False))
    else:
        og_score = get_E_P_result_P_Only(sentence=sentence, prompt=e_p_msg(few_shot=False))
    for k in word_saliency_list:
        sentences.remove(ns[k])
    ss = " ".join(sentences)
    if PE:
        new_result = get_P_E_result_P_Only(sentence=ss, prompt=p_e_msg(few_shot=False))
    else:
        new_result = get_E_P_result_P_Only(sentence=ss, prompt=e_p_msg(few_shot=False))
    og_score =  og_score[1] if og_score[0] == 1 else 1 - og_score[1]
    new_result = new_result[1] if new_result[0] == 1 else 1 - new_result[1]
    return (og_score - new_result)

def topk_sufficiency(sentence: str, word_saliency_list: list, PE: bool):
    sentences = sentence.split()
    to_sum = []
    inserted = []
    if PE:
        og_score = get_P_E_result_P_Only(sentence=sentence, prompt=p_e_msg(few_shot=False))
    else:
        og_score = get_E_P_result_P_Only(sentence=sentence, prompt=e_p_msg(few_shot=False))
    to_sum.append(og_score)
    for k in word_saliency_list:
        inserted.append(sentences[k])
    suff_sentence = list(filter(lambda a: a in inserted, sentences))
    ss = " ".join(suff_sentence)
    if PE:
        new_result = get_P_E_result_P_Only(sentence=ss, prompt=p_e_msg(few_shot=False))
    else:
        new_result = get_E_P_result_P_Only(sentence=ss, prompt=e_p_msg(few_shot=False))
    og_score =  og_score[1] if og_score[0] == 1 else 1 - og_score[1]
    new_result = new_result[1] if new_result[0] == 1 else 1 - new_result[1]
    return (og_score-new_result)

def topk_dfmit(sentence: str, word_saliency_list: list, PE: bool):
    sentences = sentence.split()
    to_sum = []
    ns = sentences.copy()
    if PE:
        og_score = get_P_E_result_P_Only(sentence=sentence, prompt=p_e_msg(few_shot=False))
    else:
        og_score = get_E_P_result_P_Only(sentence=sentence, prompt=e_p_msg(few_shot=False))
    for k in word_saliency_list:
        sentences.remove(ns[k])
    ss = " ".join(sentences)
    if PE:
        new_result = get_P_E_result_P_Only(sentence=ss, prompt=p_e_msg(few_shot=False))
    else:
        new_result = get_E_P_result_P_Only(sentence=ss, prompt=e_p_msg(few_shot=False))
    og_score =  og_score[1] if og_score[0] == 1 else 1 - og_score[1]
    new_result = new_result[1] if new_result[0] == 1 else 1 - new_result[1]
    return (new_result > 0.5 and og_score <= 0.5) or (new_result <= 0.5 and og_score > 0.5)

def main():
    dataset = load_dataset('sst', split='test')
    dataset = dataset.shuffle(seed=8)['sentence']
    dataset = dataset[:5:]
    PE_Natural_saliency_list = generating_natural_PE_saliency_list()
    EP_Natural_saliency_list = generating_natural_EP_saliency_list()
    pe_comp = []
    pe_suff = []
    pe_dfmit = []
    ep_comp = []
    ep_suff = []
    ep_dfmit = []
    for i in tqdm(range(len(dataset))):
        pe_comp.append(topk_comprehensiveness(sentence=dataset[i], word_saliency_list=PE_Natural_saliency_list[i], PE=True))
        pe_suff.append(topk_sufficiency(sentence=dataset[i], word_saliency_list=PE_Natural_saliency_list[i], PE=True))
        pe_dfmit.append(topk_dfmit(sentence=dataset[i], word_saliency_list=PE_Natural_saliency_list[i], PE=True))

        ep_comp.append(topk_comprehensiveness(sentence=dataset[i], word_saliency_list=EP_Natural_saliency_list[i], PE=False))
        ep_suff.append(topk_sufficiency(sentence=dataset[i], word_saliency_list=EP_Natural_saliency_list[i], PE=False))
        ep_dfmit.append(topk_dfmit(sentence=dataset[i], word_saliency_list=EP_Natural_saliency_list[i], PE=False))

    print("pe_comp", sum(pe_comp)/len(pe_comp))
    print("pe_suff", sum(pe_suff)/len(pe_suff))
    print("pe_dfmit", sum(pe_dfmit)/len(pe_dfmit))
    
    print("ep_comp", sum(ep_comp)/len(ep_comp))
    print("ep_suff", sum(ep_suff)/len(ep_suff))
    print("ep_dfmit", sum(ep_dfmit)/len(ep_dfmit))

if __name__ == "__main__":
    main()

    