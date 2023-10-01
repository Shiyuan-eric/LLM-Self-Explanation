from prediction import *
from datasets import load_dataset
from messages import *
import csv
import random
import pickle
from explanation import *
from tqdm import tqdm

def pickle_load_occlusion():
    file = open('PE_Occlusion_Result_Predict', 'rb')
    data = pickle.load(file=file)
    file.close()
    expl = data[1]
    print(len(expl))
    # for i in expl:
        # print(i)
    # for i in range(len(expl)):
        # print(f"There are {len(set(expl[i].values()))} saliency levels in the sentence with {len(data[0][i].split())} words.")

def generate_occlusion(dataset):
    PROMPT = few_shot_msg(few_shot=False)
    occlusion_output = []
    sentence_list = []
    for sentence in tqdm(dataset, desc="sentence processing", position=0):
        og_score = get_prediction(sentence=sentence, prompt=PROMPT)
        sentences = sentence.split()
        occlusion = {}
        for i in tqdm(range(len(sentences)), desc="occlusion processing", position=1, leave=False):
            w = sentences.pop(i)
            ss = " ".join(sentences)
            new_result = get_prediction(sentence=ss, prompt=PROMPT)
            occlusion[i] = og_score - new_result
            sentences.insert(i, w)
        occlusion = dict(sorted(occlusion.items(), key=lambda item: item[1], reverse=True))
        # print(occlusion)
        sentence_list.append(sentence)
        occlusion_output.append(occlusion)
    with open ('Occlusion_Result_Predict', 'wb') as dbfile:
        pickle.dump((sentence_list,occlusion_output), dbfile)

def generate_P_E_result(dataset):
    P_E_PROMPT = p_e_msg(few_shot=False)
    occlusion_output = []
    sentence_list = []
    for sentence in tqdm(dataset, desc="sentence processing", position=0):
        og_score = get_E_P_result_P_Only(sentence=sentence, prompt=P_E_PROMPT)
        og_score = og_score[1] if og_score[0] == 1 else (1-og_score[1])
        # print(og_score)
        sentences = sentence.split()
        occlusion = {}
        for i in tqdm(range(len(sentences)), desc="occlusion processing", position=1, leave=False):
            w = sentences.pop(i)
            ss = " ".join(sentences)
            new_result = get_E_P_result_P_Only(sentence=ss, prompt=P_E_PROMPT)
            new_result = new_result[1] if new_result[0] == 1 else (1-new_result[1])
            occlusion[i] = og_score - new_result
            sentences.insert(i, w)
        occlusion = dict(sorted(occlusion.items(), key=lambda item: item[1], reverse=True))
        # print(occlusion)
        sentence_list.append(sentence)
        occlusion_output.append(occlusion)
    with open ('PE_Occlusion_Result_Predict', 'wb') as dbfile:
        pickle.dump((sentence_list,occlusion_output), dbfile)

if __name__ == "__main__":
    dataset = load_dataset('sst', split='test')
    size = 100
    dataset = dataset.shuffle(seed=8)['sentence']
    # generate_occlusion(dataset=dataset[:size:])
    # generate_P_E_result(dataset=dataset[:size:])
    pickle_load_occlusion()