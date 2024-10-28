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
import argparse

llama_P_E_MSG = [
    {
        "role": "system",
        # "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (int classification, float confidence).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(int classification, float confidence)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"

    }
    ]

llama_E_P_MSG = [
    {
        "role": "system",
        # "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (word or punctuation, float importance). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (int classification, float confidence). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(int classification, float confidence)"
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"

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

def generate_mistral_response(messages):
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    last_instruction_idx = response.rindex("[/INST]") + 7
    return response[last_instruction_idx:]

def generate_llama_response(messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        pad_token_id = tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=False,
        # temperature=0.6,
        # top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return(tokenizer.decode(response, skip_special_tokens=True))

def parse_completion(response, sentence, PE, p_only):
    lines = response.splitlines()
    lines = [string for string in lines if string]
    lines = [string for string in lines if re.search(r'\d', string)]
    # parse the prediction and confidence
    try:
        if p_only:
            (prediction, confidence) = ast.literal_eval(lines[0])
            exp = None
        elif PE:
            exp = ast.literal_eval(lines[1])
            cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[0])
            (prediction, confidence) = ast.literal_eval(cleaned_string)
        else:
            exp = ast.literal_eval(lines[0])
            cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[1])
            (prediction, confidence) = ast.literal_eval(cleaned_string)
    except:
        if not PE:
            try:
                # Trying to see if the potential error was that there was a newline(something I saw a few times)
                cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[2])
                prediction, confidence = ast.literal_eval(cleaned_string)
            except:
                pass
        (prediction, confidence) = (0, 0.5)
    
    #parse the explanation
    try:
        if p_only:
            exp = None
        elif PE:
            exp = ast.literal_eval(lines[1])
        else:
            exp = ast.literal_eval(lines[0])
    except:
        # GPT didn't give an answer in the required format (more likely an invalid response)
        # So, make everything 0
        exp = []
        for token in sentence.split(' '):
            exp.append((token, 0.0))
        
    return (prediction, confidence, exp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program generates the explanation from SST dataset with PE or EP methods without using any additional methods')
    parser.add_argument('-pe', action='store_true', help='the boolean value that indicates whether the process is using pe(Predict and Explain)[True] or ep(Explain then Predict)[False]')
    parser.add_argument('-mistral', action='store_true', help='the boolean value that indicates if the program will use the mistral model')
    parser.add_argument('-llama', action='store_true', help='the boolean value that indicates if the program will use the llama3 model')
    args = parser.parse_args()
    
    dataset = load_dataset("sst")
    
    eval_ds = dataset["test"].shuffle(seed=8)
    dataloader = DataLoader(eval_ds, batch_size=1)
    print("loaded dataset: sst")
    
    sentences = []
    labels = []
    count = 0
    num_examples = 100
    for batch in dataloader:
        #  print(batch)
        if count == num_examples:
            break
        sentences.append(batch['sentence'][0])
        labels.append(batch['label'].item())
        count += 1
        
    PRE_PHRASE = "<review> "
    POST_PHRASE = " <review>"
    if args.mistral:
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    elif args.llama:
        MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    print(f"loaded model: {MODEL_NAME}")
    
    if args.pe:
        if args.mistral:
            messages = mistral_P_E_MSG.copy()
        elif args.llama:
            messages = llama_P_E_MSG.copy()
        print("Using PE Paradigm")
    else:
        if args.mistral:
            messages = mistral_E_P_MSG.copy()
        elif args.llama:
            messages = llama_E_P_MSG.copy()
        print("Using EP Paradigm")
    # messages = MESSAGES[:]
    responses = {}
    
    # generates response
    
    if args.mistral:
        generate_response = generate_mistral_response
    elif args.llama:
        generate_response = generate_llama_response
    
    # get completion for each sentence
    for i in tqdm(range(num_examples)):
        messages.append({"role": "user", "content": PRE_PHRASE +
                        sentences[i] + POST_PHRASE})
        completion = generate_response(messages)
        responses[sentences[i]] = completion
        messages.pop()
    # print(responses)
    # Make sure to change file names if changing from PE
    if args.pe:
        with open("llama3_response_PE.pickle", "wb") as handle:
            pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open("sentences.pickle", "wb") as handle:
            # pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("llama3_response_EP.pickle", "wb") as handle:
            pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     with open("sentences_ep.pickle", "wb") as handle:
    #         pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("true_labels.pickle", "wb") as handle:
    #         pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)