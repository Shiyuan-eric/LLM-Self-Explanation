from prediction import *
from datasets import load_dataset
from messages import *
import csv
import random
import pickle
from explanation import *
from tqdm import tqdm

og_msg = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThe movie review will be surrounded by <review> tags.\n\nExample output:\n(<int classification>, <float confidence>)"
    }
    ]

p_e_msg = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    }
    ]

e_p_msg = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
    }
    ]

def analyze_ep_result_P_only(sentence, response):
    r = response.split("\n")
    r = [i for i in r if i.strip()]
    prediction_pair = eval(r[1])
    return prediction_pair

def analyze_pe_result_P_only(sentence, response):
    r = response.split("\n")
    r = [i for i in r if i.strip()]
    prediction_pair = eval(r[0])
    return prediction_pair

def get_E_P_result_P_Only(sentence, prompt=e_p_msg):
    prompt.append({"role": "user", "content": f"<review> {sentence} <review>"})
    response = generate_response(prompt=prompt)["choices"][0]["message"]["content"]
    # print(response)
    sentence = sentence.split()
    prompt.pop()
    return analyze_ep_result_P_only(sentence, response)

def get_P_E_result_P_Only(sentence, prompt=p_e_msg):
    prompt.append({"role": "user", "content": f"<review> {sentence} <review>"})
    response = generate_response(prompt=prompt)["choices"][0]["message"]["content"]
    # print(response)
    sentence = sentence.split()
    prompt.pop()
    return analyze_pe_result_P_only(sentence, response)

def generate_E_P_result(dataset):
    E_P_PROMPT = e_p_msg
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
    with open ('EP_Occlusion_Result_Predict', 'wb') as dbfile:
        pickle.dump((sentence_list,occlusion_output), dbfile)

def generate_P_E_result(dataset):
    P_E_PROMPT = p_e_msg
    occlusion_output = []
    sentence_list = []
    for sentence in tqdm(dataset, desc="sentence processing", position=0):
        og_score = get_P_E_result_P_Only(sentence=sentence, prompt=P_E_PROMPT)
        og_score = og_score[1] if og_score[0] == 1 else (1-og_score[1])
        # print(og_score)
        sentences = sentence.split()
        occlusion = {}
        for i in tqdm(range(len(sentences)), desc="occlusion processing", position=1, leave=False):
            w = sentences.pop(i)
            ss = " ".join(sentences)
            new_result = get_P_E_result_P_Only(sentence=ss, prompt=P_E_PROMPT)
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

    generate_P_E_result(dataset=dataset[:size:])
    generate_E_P_result(dataset=dataset[:size:])
    # pickle_load_occlusion()