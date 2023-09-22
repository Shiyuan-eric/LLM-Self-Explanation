from prediction import *
from datasets import load_dataset
from messages import *
import csv
import random

PROMPT = few_shot_msg(few_shot=True)

def explain_occlusion(sentence, og_score):
    sentences = sentence.split()
    occlusion = {}
    for i in range(len(sentences)):
        w = sentences.pop(i)
        ss = " ".join(sentences)
        # print(ss)
        new_result = float(get_prediction(sentence=ss, prompt=PROMPT))
        occlusion[i] = og_score - new_result
        sentences.insert(i, w)
    occlusion = dict(sorted(occlusion.items(), key=lambda item: item[1], reverse=True))
    # print(occlusion)
    # word_occlusion_saliency = {}
    # ns = sentence.split()
    # for k,v in occlusion.items():
    #     print(k,v)
    #     # word_occlusion_saliency.append(ns[k])
    #     word_occlusion_saliency[ns[k]] = v
    # word_occlusion_saliency = dict(sorted(word_occlusion_saliency.items(), key=lambda item: item[1], reverse=True))
    # print(word_occlusion_saliency)
    return occlusion

if __name__ == "__main__":
    dataset = load_dataset('sst', split='test')['sentence']
    size = 10
    csvfile = open("rand_occlusion_attr.csv", 'w')
    csvwriter = csv.writer(csvfile)
    data = []
    random.seed(10)
    data = random.shuffle(dataset)[:size:]
    # for _ in range(size):
    #     t = random.randint(0, len(dataset))
    #     if t not in data:
    #         data.append(t)
    # print(data)
    
    for i in data:
        print(i)
        og_score = get_prediction(sentence=i, prompt=PROMPT)
        v = explain_occlusion(sentence=i, og_score=og_score)
        print(v)
        csvwriter.writerow([i, dataset[i], og_score, v])

    csvfile.close()