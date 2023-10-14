import pickle
from datasets import load_dataset
import random
import numpy as np
import matplotlib.pyplot as plt

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
    # PE_word_saliency_list, PE_attribute_val = generating_PE_word_saliency_list()
    # EP_word_saliency_list, EP_attribute_val= generating_EP_word_saliency_list()
    PE_Occlusion_word_saliency_list, PE_Occlusion_attribute_val = generating_PE_Occlusion_saliency_list()
    # EP_Occlusion_word_saliency_list, EP_Occlusion_attribute_val = generating_EP_Occlusion_saliency_list()
    # PE_LIME_word_saliency_list, PE_LIME_attribute_val = generating_PE_LIME_saliency_list(dataset)
    # EP_LIME_word_saliency_list, EP_LIME_attribute_val = generating_EP_LIME_saliency_list(dataset)
    # PE_Natural_saliency_list = generating_natural_PE_saliency_list()
    EP_Natural_saliency_list = generating_natural_EP_saliency_list()
    print(PE_Occlusion_attribute_val)

def process():
    fn = 'gpt_response_EP.pickle'
    d = pickle.load(open(fn, 'rb'))
    ep_sent = 'Never engaging , utterly predictable and completely void of anything remotely interesting or suspenseful .'
    idx = list(d.keys()).index(ep_sent)
    ep_vals = d[ep_sent]
    ep_vals = eval(ep_vals.split('\n')[0])
    _, ep_vals = map(list, zip(*ep_vals))
    print(ep_vals)

    fn = 'EP_Occlusion_Result_Predict'
    d = pickle.load(open(fn, 'rb'))
    ep_occl_vals = [d[1][idx][i] for i in range(15)]
    print(ep_occl_vals)

    fn = 'LIME_response_EP_0_100.pickle'
    d = pickle.load(open(fn, 'rb'))
    for exp in d:
        if ' '.join(exp[1]) == ep_sent:
            break
    exp_dict = {e[0]: e[1][0] for e in exp[0]}
    ep_lime_vals = [exp_dict[w] for w in ep_sent.split(' ')]
    print(ep_lime_vals)

    fn = 'parse_topk_EP.pickle'
    d = pickle.load(open(fn, 'rb'))
    words = d[ep_sent][0]
    ep_topk_idxs = [int(w in words) for w in ep_sent.split()]
    print(ep_topk_idxs)

    assert len(ep_vals)==len(ep_occl_vals)==len(ep_lime_vals)==len(ep_topk_idxs)


    fn = 'gpt_response_PE.pickle'
    d = pickle.load(open(fn, 'rb'))
    pe_sent = "( A ) superbly controlled , passionate adaptation of Graham Greene 's 1955 novel ."
    idx = list(d.keys()).index(pe_sent)

    pe_vals = d[pe_sent]
    pe_vals = eval(pe_vals.split('\n')[1])
    _, pe_vals = map(list, zip(*pe_vals))
    pe_vals.insert(pe_vals.index(0.6), 0.6)
    print(pe_vals)

    fn = 'PE_Occlusion_Result_Predict'
    d = pickle.load(open(fn, 'rb'))
    pe_occl_vals = [d[1][idx][i] for i in range(15)]
    print(pe_occl_vals)

    fn = 'LIME_explanations_PE.pickle'
    d = pickle.load(open(fn, 'rb'))
    for exp in d:
        if ' '.join(exp[1]) == pe_sent:
            break
    exp_dict = {e[0]: e[1][0] for e in exp[0]}
    pe_lime_vals = [exp_dict[w] for w in pe_sent.split(' ')]
    print(pe_lime_vals)

    fn = 'parse_topk_PE.pickle'
    d = pickle.load(open(fn, 'rb'))
    words = d[pe_sent][0]
    pe_topk_idxs = [int(w in words) for w in pe_sent.split()]

    assert len(pe_vals)==len(pe_occl_vals)==len(pe_lime_vals)==len(pe_topk_idxs)

    plt.figure(figsize=[10, 3.5])
    
    plt.subplot(2, 1, 1)
    xs = np.arange(15)
    plt.bar(xs-0.2, ep_occl_vals, color='C0', width=0.2, label='E-P Occlusion')
    plt.bar(xs, ep_lime_vals, color='C1', width=0.2, label='E-P LIME')
    plt.bar(xs+0.2, ep_vals, color='C2', width=0.2, label='E-P SelfExp')
    xmin, xmax, ymin, ymax = plt.axis()
    for i in range(0, 14):
        plt.plot([i+0.5, i+0.5], [ymin, ymax], 'C7--', lw=0.5)
    plt.axis([-0.5, 14.5, ymin, ymax])
    plt.legend(loc='lower right', fontsize=8)
    plt.xticks(xs, ep_sent.split(), rotation=0, ha='center', fontsize=8)

    plt.subplot(2, 1, 2)
    xs = np.arange(15)
    plt.bar(xs-0.2, pe_occl_vals, color='C0', width=0.2, label='P-E Occlusion')
    plt.bar(xs, pe_lime_vals, color='C1', width=0.2, label='P-E LIME')
    plt.bar(xs+0.2, pe_vals, color='C2', width=0.2, label='P-E SelfExp')
    xmin, xmax, ymin, ymax = plt.axis()
    for i in range(0, 14):
        plt.plot([i+0.5, i+0.5], [ymin, ymax], 'C7--', lw=0.5)
    plt.axis([-0.5, 14.5, ymin, ymax])
    plt.legend(loc='upper right', fontsize=8)
    pe_words = pe_sent.split()
    plt.xticks(xs, pe_words, rotation=0, ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('exp_visualization.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    process()

