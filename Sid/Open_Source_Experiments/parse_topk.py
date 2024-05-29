import time
import os
import sys
import getopt
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import argparse


random.seed(0)
random_range = 10e-4

def generate_response():
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    last_instruction_idx = response.rindex("[/INST]") + 7
    return response[last_instruction_idx:]


def topk_prompt(k, pe):
    prompt = "As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word and punctuation in determining the overall positivity level. Your task is to identify the top %d most significant words, ranked from the most positive sentiment to the least positive sentiment. Additionally, you need to determine whether the movie review is positive or negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nThe movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review's content.\n\nOnly output the list of %d words in the form of a comma separated list with spaces, and the prediction(as a number) and confidence following after, nothing more. Do you understand?" % (
        k, k)
    ep_prompt = "As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word and punctuation in determining the overall positivity level. Your task is to identify the top %d most significant words, ranked from the most positive sentiment to the least positive sentiment. Additionally, you need to determine whether the movie review is positive or negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Please note that the coherence of the sentence is not relevant; your focus should be on analyzing the sentiment.\n\nThe movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review's content.\n\nOnly output the list of %d words in the form of a comma separated list, with the prediction(as a number) and confidence following after, nothing more." %(k, k)
    pe_prompt = "As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word and punctuation in determining the overall positivity level. Your task is to determine whether the movie review is positive or negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. In addition, you need to identify the top %d most significant words, ranked from the most positive sentiment to the least positive sentiment. Please note that the coherence of the sentence is not relevant; your focus should be on analyzing the sentiment.\n\nThe movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review's content.\n\nOnly output the prediction(as a number) and confidence, with the list of %d words in the form of a comma separated list following after, nothing more." % (k, k)
    if pe:
        return pe_prompt
    else:
        return ep_prompt
    # return prompt

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
        except:
            continue
    
    return final_expl, result_tuple


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The program parse_topk.py generates the explanation with using topk output format.')
    parser.add_argument('-pe', action='store_true', help='the boolean value that indicates whether the process is using pe(Predict and Explain)[True] or ep(Explain then Predict)[False]')
    args = parser.parse_args()
    
    PRE_PHRASE = "<review> "
    POST_PHRASE = " <review>"
    
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(device)
    
    
    dataset = load_dataset("sst")
    
    eval_ds = dataset["test"].shuffle(seed=8)
    dataloader = DataLoader(eval_ds, batch_size=1)
    
    print("loaded dataset and device")
    
    
    sentences = []
    model_labels = []
    count = 0
    num_examples = 100
    for batch in dataloader:
        if count == num_examples:
            break
        sentences.append(batch['sentence'][0])
        count += 1
    
    messages = []
    responses = []
    
    
    
    for i in tqdm(range(num_examples)):
        k = int(len(sentences[i].split(" ")) * 0.2)
        if k < 1:
            k = 1
        messages.append({"role": "user", "content": topk_prompt(k, args.pe)})
        messages.append(
            {"role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."})
        messages.append({"role": "user", "content": PRE_PHRASE +
                        sentences[i] + POST_PHRASE})
        completion = generate_response()
        expl, label = parse_topk(completion, k, sentences[i])
        responses.append(expl)
        model_labels.append(label)
        print(expl)
        print(label)
        messages.pop()
        messages.pop()
        messages.pop()
    
    # print(responses)
    if args.pe:
        with open("topk_expl_PE.pickle", "wb") as handle:
            pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("topk_labels_PE.pickle", "wb") as handle:
            pickle.dump(model_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("topk_expl_EP.pickle", "wb") as handle:
            pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("topk_labels_EP.pickle", "wb") as handle:
            pickle.dump(model_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
