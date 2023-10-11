import pickle
from datasets import load_dataset
import random

random_range = 10e-4
random.seed(0)

def reconstruct_explaination(response: list, sentence: list):
    global ratio
    word_list = []
    expl_list = []
    attribution_value ={}
    for i in sentence:
        if len(response) == 0:
            break
        elif i == response[0][0]:
            word_list.append(i)
            expl_list.append(response[0][1])
            response.pop(0)
        else:
            while i in list(dict(response).keys()):
                response.pop(0)
                if i == response[0][0]:
                    word_list.append(i)
                    expl_list.append(response[0][1])
                    response.pop(0)
                    break
    diff = [x for x in sentence if x not in word_list]
    for j in range(len(sentence)):
        if len(word_list) != 0 and sentence[j] == word_list[0]:
            attribution_value[j] = expl_list[0]+ random.uniform(-1 * random_range, random_range)
            word_list.pop(0)
            expl_list.pop(0)
        else:
            attribution_value[j] = 0.0 + random.uniform(-1 * random_range, random_range)
    return attribution_value

def analyze_ep_result(sentence, response):
    r = response.split("\n")
    r = [i for i in r if i.strip()]
    prediction_pair = eval(r[1])
    attribution_value = eval(r[0])
    final_attribution_value = reconstruct_explaination(attribution_value, sentence)
    final_attribution_value = dict(sorted(final_attribution_value.items(), key=lambda item: item[1], reverse=True))
    return prediction_pair, final_attribution_value

def analyze_pe_result(sentence, response):
    r = response.split("\n")
    r = [i for i in r if i.strip()]
    prediction_pair = eval(r[0])
    attribution_value = eval(r[1])
    final_attribution_value = reconstruct_explaination(attribution_value, sentence)
    final_attribution_value = dict(sorted(final_attribution_value.items(), key=lambda item: item[1], reverse=True))
    return prediction_pair, final_attribution_value

def generating_PE_word_saliency_list():
    filename="gpt_response_PE.pickle"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)
    new_expl = []
    val = []
    for sentence, response in expl.items():
        new_expl.append(list(analyze_pe_result(sentence.split(), response)[1].keys()))
        val.append(list(analyze_pe_result(sentence.split(), response)[1].values()))
    return new_expl, val

def generating_EP_word_saliency_list():
    filename="gpt_response_EP.pickle"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)
    
    new_expl = []
    val = []
    for sentence, response in expl.items():
        new_expl.append(list(analyze_ep_result(sentence.split(), response)[1].keys()))
        val.append(list(analyze_ep_result(sentence.split(), response)[1].values()))
    return new_expl, val

def generating_PE_Occlusion_saliency_list():
    filename = "PE_Occlusion_Result_Predict"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)

    new_expl = []
    val = []
    for response in expl[1]:
        for k, v in response.items():
            response[k] = v + random.uniform(-1 * random_range, random_range)
        new_expl.append(list(dict(sorted(response.items(), key=lambda item: item[1], reverse=True)).keys()))
        val.append(list(dict(sorted(response.items(), key=lambda item: item[1], reverse=True)).values()))
    return new_expl, val

def generating_EP_Occlusion_saliency_list():
    filename = "EP_Occlusion_Result_Predict"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)

    new_expl = []
    val = []
    for response in expl[1]:
        for k, v in response.items():
            response[k] = v + random.uniform(-1 * random_range, random_range)
        new_expl.append(list(dict(sorted(response.items(), key=lambda item: item[1], reverse=True)).keys()))
        val.append(list(dict(sorted(response.items(), key=lambda item: item[1], reverse=True)).values()))

    return new_expl, val

def generating_PE_LIME_saliency_list(dataset):
    dataset = dataset[:100:]
    filename = "LIME_explanations_PE.pickle"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)
    d = {}
    for i in expl:
        d[dataset.index(" ".join(i[1]))] = i[0]
    d = dict(sorted(d.items()))
    new_expl = []
    val = []
    # print(d)
    for i in d.values():
        temp_expl = []
        temp_val = []
        for j in i:
            temp_expl.append(j[1][1])
            temp_val.append(j[1][0])
        new_expl.append(temp_expl)
        val.append(temp_val)

    return new_expl, val

def generating_EP_LIME_saliency_list(dataset):
    dataset = dataset[:100:]
    filename = "LIME_response_EP_0_100.pickle"
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)
    d = {}
    for i in expl:
        d[dataset.index(" ".join(i[1]))] = i[0]
    d = dict(sorted(d.items()))
    new_expl = []
    val = []
    # print(d)
    for i in d.values():
        temp_expl = []
        temp_val = []
        for j in i:
            temp_expl.append(j[1][1])
            temp_val.append(j[1][0])
        new_expl.append(temp_expl)
        val.append(temp_val)

    return new_expl, val

def generating_natural_PE_saliency_list():
    filename = "parse_topk_PE.pickle"
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

def main():
    size = 100
    dataset = load_dataset('sst', split='test')
    dataset = dataset.shuffle(seed=8)['sentence']
    PE_word_saliency_list, PE_attribute_val = generating_PE_word_saliency_list()
    EP_word_saliency_list, EP_attribute_val= generating_EP_word_saliency_list()
    PE_Occlusion_word_saliency_list, PE_Occlusion_attribute_val = generating_PE_Occlusion_saliency_list()
    EP_Occlusion_word_saliency_list, EP_Occlusion_attribute_val = generating_EP_Occlusion_saliency_list()
    PE_LIME_word_saliency_list, PE_LIME_attribute_val = generating_PE_LIME_saliency_list(dataset)
    EP_LIME_word_saliency_list, EP_LIME_attribute_val = generating_EP_LIME_saliency_list(dataset)
    PE_Natural_saliency_list = generating_natural_PE_saliency_list()
    EP_Natural_saliency_list = generating_natural_EP_saliency_list()

if __name__ == "__main__":
    main()