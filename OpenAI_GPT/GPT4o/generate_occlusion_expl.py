from scipy import stats
import time
import os
import sys
import getopt
import torch
import pickle
import random
import re
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
# from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import pandas as pd

from generate_model_expl import generate_response

gpt4o_PE_MSG = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
            }
        ]
    }
]

gpt4o_EP_MSG = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
            }
        ]
    }
]

gpt4o_P_MSG = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)"
            }
        ]
    }
]

def loadData(filename):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
def storeData(filename, data):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

class OcclusionExplanationGenerator():
    def __init__(self, PE, dataset, noise=True, p_only=None):
        # self.messages = messages
        self.fails = 0
        self.total = 1
        # self.ovr_fails = 0
        # self.ovr_total = 1
        self.PE = PE
        self.dataset = dataset
        self.noise = noise
        self.p_only = p_only

        if self.p_only:
            print("In p_only Mode.")
            self.cache_name = 'gpt4o_p_cache.pickle'
        elif self.PE:
            self.cache_name = 'gpt4o_pe_cache.pickle'
        else:
            print("In EP Mode.")
            self.cache_name = 'gpt4o_ep_cache.pickle'

        self.pre_phrase = "<review> "
        self.post_phrase = " <review>"

        self.random_range = 10e-4
        random.seed(0)

    def init_message(self):
        if self.p_only:
            self.messages = gpt4o_P_MSG.copy()
        elif self.PE:
            self.messages = gpt4o_PE_MSG.copy()
        else:
            self.messages = gpt4o_EP_MSG.copy()

    def trim_string(self, s):
        start_index = s.find('[')
        end_index = s.rfind(']')

        if start_index != -1 and end_index != -1 and start_index < end_index:
            if self.PE:
                pre_start_index = s[:start_index].find('(')
                pre_end_index = s[:start_index].rfind(')')
                return s[0:start_index][pre_start_index:pre_end_index+1], s[start_index:end_index + 1].replace("\n", "")
            else:
                pre_start_index = s[end_index+1:].find('(')
                pre_end_index = s[end_index+1:].rfind(')')
                return s[start_index:end_index + 1].replace("\n", ""), s[end_index+1: -1][pre_start_index:pre_end_index+1]
        else:
            return s

    def parse_completion(self, response):
        if self.p_only:
            lines = response.splitlines()
        else:
            lines = self.trim_string(response)
        lines = [string for string in lines if string]
        lines = [string for string in lines if re.search(r'\d', string)]
        self.total += 1
        try:
            if self.p_only:
                cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[0])
                (prediction, confidence) = ast.literal_eval(cleaned_string)
            elif self.PE:
                cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[0])
                (prediction, confidence) = ast.literal_eval(cleaned_string)
            else:
                cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[1])
                (prediction, confidence) = ast.literal_eval(cleaned_string)
        except:
            if not self.PE:
                try:
                    # Trying to see if the potential error was that there was a newline(something I saw a few times)                    
                    cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[2])
                    prediction, confidence = ast.literal_eval(cleaned_string)
                    return (prediction, confidence, None)
                except:
                    pass
            # GPT didn't give an answer in the required format (more likely an invalid response)
            # So, make everything 0
            (prediction, confidence) = (0, 0.5)
            self.fails += 1
        return (prediction, confidence, None)

    def get_completion(self, message):
        model_response = generate_response(message)
        prediction, confidence, _ = self.parse_completion(model_response)
        return prediction, confidence


    def get_result(self, phrase):
        try:
            cache_dict = loadData(self.cache_name)
        except:
            cache_dict = dict()
        if phrase in cache_dict:
            label, prob = cache_dict[phrase]
        else:
            self.init_message()
            self.messages.append({"role": "user", "content": [{"type": "text", "text": 
                self.pre_phrase + phrase + self.post_phrase}]})
            label, prob = self.get_completion(self.messages)
            cache_dict[phrase] = (label, prob)
            storeData(self.cache_name, cache_dict)
        if label == 1:
            score = prob
        else:
            score = 1-prob
        return score

    def compute_occlusion_saliency(self):
        # Model_Evaluator.get_completion()
        # if self.PE: # TODO: What's the new prompt for the open_source models?
        #     msg = P_E_MSG
        # else:
        #     msg = E_P_MSG
        occlusion_output = []
        sentence_list = []
        count = 0
        output = list()
        for sentence in tqdm(self.dataset, desc="sentence processing", position=0):
        # for sentence in self.dataset:
            # full_msg = self.generate_whole_prompt(sentence)
            og_score = self.get_result(sentence)
            # print("The original score is:", og_score)
            sentences = sentence.split()
            ns =sentence.split()
            
            occlusion = {}
            for i in tqdm(range(len(sentences)), desc="occlusion processing", position=1, leave=False):
                w = sentences.pop(i)
                ss = " ".join(sentences)
                # full_new_msg = self.generate_whole_prompt(ss)
                new_result = self.get_result(ss)
                # print("\nThe new score is:", new_result)
                if self.noise:
                    occlusion[i] =  og_score - new_result + random.uniform(0, self.random_range)
                else:
                    occlusion[i] =  og_score - new_result
                # print(f"The occlusion for {ns[i]} is {occlusion[i]}")
                sentences.insert(i, w)
            occlusion = dict(sorted(occlusion.items(), key=lambda item: item[1], reverse=True))
            sentence_list.append(sentence)
            # print(f"\n Occlusion = {occlusion}")
            occlusion_output.append(occlusion)
            attr = list()
            for k, v in occlusion.items():
                attr.append((sentences[k], (v, k)))
            # print(attr)
            # print(attr)
            output.append((attr, sentences))

        if self.p_only:
            storeData('gpt4o_p_occlusion.pickle', output)
        elif self.PE:
            storeData('gpt4o_pe_occlusion.pickle', output)
        else:
            storeData('gpt4o_ep_occlusion.pickle', output)
        

if __name__ == "__main__":
    dataset = load_dataset("sst")
    eval_ds = dataset["test"].shuffle(seed=8)
    
    print("\n *** dataset has been loaded *** \n")
    NUM_ITER = 100
    # sample = eval_ds['sentence'][2]


    # # PE
    # occlusion_generator = OcclusionExplanationGenerator(PE=True, dataset = eval_ds['sentence'][:NUM_ITER])
    # occlusion_generator.compute_occlusion_saliency()

    # # #EP
    # occlusion_generator = OcclusionExplanationGenerator(PE=False, dataset = eval_ds['sentence'][:NUM_ITER])
    # occlusion_generator.compute_occlusion_saliency()

    # P_Only
    occlusion_generator = OcclusionExplanationGenerator(p_only = True, PE=False, dataset = eval_ds['sentence'][:NUM_ITER])
    occlusion_generator.compute_occlusion_saliency()