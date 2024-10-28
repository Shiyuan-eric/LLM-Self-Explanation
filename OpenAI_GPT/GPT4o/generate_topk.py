import os
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
import lime
from lime.lime_text import IndexedString
from lime.lime_text import LimeTextExplainer
from lime import lime_text
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
# from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from datasets import load_dataset
 

from lime_explanation_generator import LimeExplanationGenerator
import argparse
import re
from generate_model_expl import generate_response

random.seed(0)
random_range = 10e-4

def topk_prompt(k, pe):
    ep_prompt = "As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word and punctuation in determining the overall positivity level. Your task is to identify the top %d most significant words, ranked from the most positive sentiment to the least positive sentiment. Additionally, you need to determine whether the movie review is positive or negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Please note that the coherence of the sentence is not relevant; your focus should be on analyzing the sentiment.\n\nThe movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review's content.\n\nOnly output the list of %d words in the form of a comma separated list, with the prediction(as a number) and confidence following after, nothing more." %(k, k)
    pe_prompt = "As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word and punctuation in determining the overall positivity level. Your task is to determine whether the movie review is positive or negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. In addition, you need to identify the top %d most significant words, ranked from the most positive sentiment to the least positive sentiment. Please note that the coherence of the sentence is not relevant; your focus should be on analyzing the sentiment.\n\nThe movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review's content.\n\nOnly output the prediction(as a number) and confidence, with the list of %d words in the form of a comma separated list following after, nothing more." % (k, k)
    if pe:
        return pe_prompt
    else:
        return ep_prompt

def generate_random_float_list(length):
    random_numbers = [random.uniform(0, random_range) for _ in range(length)]
    # Sort the list in descending order
    random_numbers.sort(reverse=True)
    return random_numbers

def extract_remaining(l1, l2):
    return [elem for elem in l2 if elem not in l1]

def generate_expl(index_list, words):
    random_float_list = generate_random_float_list(len(words))
    iterator = iter(random_float_list)
    remaining_list = extract_remaining(index_list, list(range(len(words))))
    random.shuffle(remaining_list)
    final_expl_list = index_list+remaining_list
    explanation = []
    for i in final_expl_list:
        explanation.append((words[i], (next(iterator), i)))
    return (explanation, words)

def parse_topk(response, k, sentence):
    print("response:", response)
    while True:
        result_tuple = input(f"\nEnter the prediction result in the format of (prediction,likelyhood):\n")
        try:
            result_tuple = eval(result_tuple)
            break
        except:
            continue
        
    
    words = sentence.split()
    word_dict = [f"{word}-{index}" for index, word in enumerate(words)]

    while True:
        print(f"\nsentence word_index representation:\n{word_dict}")
        try:
            index_str = input(f"\nEnter the index of the top {k} words in the format of i1,i2,i3:\n")
            temp_list = index_str.split(',')
            index_list = [int(x) for x in temp_list]
            final_expl = generate_expl(index_list, words)
            break
        except Exception as error:
            print(error) 
            continue
    
    return final_expl, result_tuple

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-pe', action='store_true', help='')

    args = parser.parse_args()

    PRE_PHRASE = "<review> "
    POST_PHRASE = " <review>"

    dataset = load_dataset("sst")
    eval_ds = dataset["test"].shuffle(seed=8)
    NUM_ITER = 100
    sentences = eval_ds['sentence'][:NUM_ITER]
    responses = []
    model_labels = []
    for i in tqdm(range(NUM_ITER)):
        k = int(len(sentences[i].split(" ")) * 0.2)
        if k < 1:
            k = 1
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{topk_prompt(k, args.pe)}"
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": PRE_PHRASE + sentences[i] + POST_PHRASE
                    }
                ]
            }
        ]
        completion = generate_response(messages)
        expl, label = parse_topk(completion, k, sentences[i])
        responses.append(expl)
        model_labels.append(label)
        print(expl)
        print(label)

    if args.pe:
        with open("gpt4o_topk_expl_PE.pickle", "wb") as handle:
            pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("gpt4o_topk_labels_PE.pickle", "wb") as handle:
            pickle.dump(model_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("gpt4o__topk_expl_EP.pickle", "wb") as handle:
            pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("gpt4o_topk_labels_EP.pickle", "wb") as handle:
            pickle.dump(model_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)