from occlusion import *
from datasets import load_dataset
import statistics
from messages import *
import csv
import ast
from scipy import stats

PROMPT = few_shot_msg(few_shot=False)

def count_unique_values(d):
    return len(set(d.values()))

def k_largest(d, k): 
    values = list(set(d.values()))
    values.sort(reverse=True)
    if k <= len(values):
        kth_value = values[k]
    else:
        return "k is larger than the number of unique values in the dictionary"
    return [key for key, value in d.items() if value == kth_value]

def re_organize(d): # exp += np.random.random(len(explanation))*1e-5
    word_saliency_list = []
    for k in range(count_unique_values(d=d)):
        # t = set(k_largest(d,k).keys())
        random.shuffle(t := list(k_largest(d,k).keys()))
        word_saliency_list += t
    return word_saliency_list

def string_to_dict(dict_string):
    return(ast.literal_eval(dict_string))

def evaluate_comprehensiveness(sentence: str, word_saliency_list: list):
    sentences = sentence.split()
    to_sum = []
    ns = sentences.copy()
    fail = 0
    og_score, _ = get_prediction(sentence=sentence, prompt=PROMPT)
    for k in word_saliency_list:
        sentences.remove(ns[k])
        ss = " ".join(sentences)
        new_result, notsucceed = get_prediction(sentence=ss, prompt=PROMPT)
        to_sum.append(og_score-new_result)
        if notsucceed:
            fail += 1
    Sum = sum(to_sum)
    print(f"There are {fail} number of failures in {len(ns)} words.")
    return (Sum/(1+len(to_sum)))

def evaluate_sufficiency(sentence: str, word_saliency_list: list):
    sentences = sentence.split()
    to_sum = []
    inserted = []
    og_score = get_prediction(sentence=sentence, prompt=PROMPT)
    to_sum.append(og_score)
    for k in word_saliency_list:
        inserted.append(sentences[k])
        suff_sentence = list(filter(lambda a: a in inserted, sentences))
        ss = " ".join(suff_sentence)
        new_result = get_prediction(sentence=ss, prompt=PROMPT)
        to_sum.append(og_score-new_result)
    Sum = sum(to_sum)
    return (Sum/(len(to_sum)))

def evaluate_dfmit(sentence: str, word_saliency_list: list):
    sentences = sentence.split()
    ns = sentences.copy()
    og_score = get_prediction(sentence=sentence, prompt=PROMPT)
    # for i in range(count_unique_values(attribution_values)):
    # k = k_largest(word_saliency_list, 0)
    # result = []
    # for j in k:
    sentences.remove(ns[word_saliency_list[0]])
    ss = " ".join(sentences)
    result = get_prediction(sentence=ss, prompt=PROMPT)
    #     print(result)
    #     sentences.insert(j, ns[j]) # Like the Occlusion Function
    # new_result = sum(result)/len(result)
    return (result >= 0.5 and og_score < 0.5) or (result < 0.5 and og_score >= 0.5)

def evaluate_dffrac(sentence: str, word_saliency_list: list):
    sentences = sentence.split()
    ns = sentences.copy()
    og_score = get_prediction(sentence=sentence, prompt=PROMPT)
    total = 0
    for k in word_saliency_list:
        sentences = list(filter(lambda a: a != ns[k], sentences))
        ss = " ".join(sentences)
        new_result = get_prediction(sentence=ss, prompt=PROMPT)
        if (new_result >= 0.5 and og_score >= 0.5) or (new_result < 0.5 and og_score < 0.5):
            total += 1
        else:
            total += 1
            break
    return (total/len(ns))
    
def evaluate_del_rank_correlation(sentence: str, word_saliency_list: list):
    sentences = sentence.split()
    ns = sentences.copy()
    result = []
    og_score = get_prediction(sentence=sentence, prompt=PROMPT)
    for i in word_saliency_list:
        sentences.remove(ns[i])
        ss = " ".join(sentences)
        new_score = get_prediction(sentence=ss, prompt=PROMPT)
        result.append(og_score-new_score)
        sentences.insert(i, ns[i])
    rankdel = stats.spearmanr(result, sentences) # spearmanr(result, explanation)??
    return rankdel

def feature_agreement(word_saliency_list_1: list, word_saliency_list_2: list, k: int):
    top_k_1 = word_saliency_list_1[:k:]
    top_k_2 = word_saliency_list_2[:k:]
    result = [i == j for i in top_k_1 for j in top_k_2]
    return result.count(1) / k

def rank_agreement(word_saliency_list_1: list, word_saliency_list_2: list, k: int):
    top_k_1 = word_saliency_list_1[:k:]
    top_k_2 = word_saliency_list_2[:k:]
    result = []
    for i in top_k_1:
        result.append(i in top_k_2 and top_k_1.index(i) == top_k_2.index(i))
    return result.count(1) / k
    
def rank_correlation(word_saliency_list_1: list, word_saliency_list_2: list, features: set):
    ranking_1 = []
    ranking_2 = []
    for i in features:
        ranking_1.append(word_saliency_list_1)
        ranking_2.append(word_saliency_list_2)




if __name__ == "__main__":
    dataset = load_dataset('sst', split='test')['sentence']
    occlusion_comprehensiveness = []
    occlusion_sufficiency = []
    occlusion_dfmit = []
    occlusion_dffrac = []
    with open(f"rand_occlusion_attr.csv", "r") as openfile:
        csvFile = csv.reader(openfile)
        for lines in csvFile:
            index, sentence, occlusion_attribution_values = lines
            occlusion_attribution_values = string_to_dict(occlusion_attribution_values)
            occlusion_comprehensiveness.append(evaluate_comprehensiveness(sentence=sentence, word_saliency_list=occlusion_attribution_values))
            # occlusion_sufficiency.append(evaluate_sufficiency(sentence=sentence, word_saliency_list=occlusion_attribution_values))
            # occlusion_dfmit.append(evaluate_dfmit(sentence=sentence, word_saliency_list= list(occlusion_attribution_values.keys())))
            # occlusion_dffrac.append(evaluate_dffrac(sentence=sentence, word_saliency_list=occlusion_attribution_values))
    print("Occlusion comprehensiveness:", statistics.mean(occlusion_comprehensiveness))
    # print("Occlusion sufficiency:", statistics.mean(occlusion_sufficiency))
    # print("Occlusion df_mit:",statistics.mean(occlusion_dfmit))
    # print("Occlusion df_frac:",statistics.mean(occlusion_dffrac))
