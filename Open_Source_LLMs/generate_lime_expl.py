# LIME
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
from transformers import AutoTokenizer, AutoModelForCausalLM

from lime_explanation_generator import LimeExplanationGenerator
import argparse
import re


llama_P_MSG = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)"
    }
    ]

llama_P_E_MSG = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    }
    ]

llama_E_P_MSG = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
    }
    ]

mistral_P_MSG = [
    {
        "role": "user",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
    ]

mistral_P_E_MSG = [
    {
        "role": "user",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
    ]
    
mistral_E_P_MSG = [
    {
        "role": "user",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The LIME Explanation Generator')
    parser.add_argument('-pe', action='store_true', help='the boolean value that indicates whether the process is using pe(Predict and Explain)[True] or ep(Explain then Predict)[False]')
    parser.add_argument('-mistral', action='store_true', help='the boolean value that indicates if the program will use the mistral model')
    parser.add_argument('-llama', action='store_true', help='the boolean value that indicates if the program will use the llama3 model')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mistral:
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    elif args.llama:
        MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(device)
    print("model loaded", MODEL_NAME, device)
    
    max_tokens = 2048
    
    
    if args.pe:
        print("Doing PE paradiam")
        if args.mistral:
            P_E_MSG = mistral_P_E_MSG
        elif args.llama:
            P_E_MSG = llama_P_E_MSG
        start = 0
        end = 100
        batch_size = 10  # checkpoints every batch_size batches into a pickle file
        for i in tqdm(range(start, end, batch_size)):
            lime_generator = LimeExplanationGenerator(
                response_filename='sentences.pickle', model=model, tokenizer=tokenizer, max_tokens=max_tokens, PE=True, start=i, end=i+batch_size, mistral=args.mistral, llama=args.llama)
            lime_generator.compute_lime_saliency([0, 1])
            with open("llama_LIME_response_PE_%d_%d.pickle" % (i, i+batch_size), "wb") as handle:
                pickle.dump(lime_generator.explanations, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        prefixed = [filename for filename in os.listdir(
            '.') if filename.startswith("llama_LIME_response_PE_")]
        prefixed.sort()
        explanations = []
        for filename in prefixed:
            with open(filename, 'rb') as handle:
                cur = pickle.load(handle)
            for e in cur:
                explanations.append(e)
        with open("llama_LIME_response_PE_%d_%d.pickle" % (0, end), "wb") as handle:
            pickle.dump(explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Doing EP paradiam")
        if args.mistral:
            E_P_MSG = mistral_E_P_MSG
        elif args.llama:
            E_P_MSG = llama_E_P_MSG
        start = 30
        end = 100
        batch_size = 10  # checkpoints every batch_size batches into a pickle file
        # start, end, batch_size = 0,1,1
        for i in tqdm(range(start, end, batch_size)):
            lime_generator = LimeExplanationGenerator(
                response_filename='sentences.pickle', model=model, tokenizer=tokenizer, max_tokens=max_tokens, PE=False, start=i, end=i+batch_size, mistral=args.mistral, llama=args.llama)
            lime_generator.compute_lime_saliency([0, 1])
            with open("llama_LIME_response_EP_%d_%d.pickle" % (i, i+batch_size), "wb") as handle:
                pickle.dump(lime_generator.explanations, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        prefixed = [filename for filename in os.listdir(
            '.') if filename.startswith("llama_LIME_response_EP_")]
        prefixed.sort()
        explanations = []
        for filename in prefixed:
            with open(filename, 'rb') as handle:
                cur = pickle.load(handle)
            for e in cur:
                explanations.append(e)
        with open("llama_LIME_response_EP_%d_%d.pickle" % (0, end), "wb") as handle:
            pickle.dump(explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)
