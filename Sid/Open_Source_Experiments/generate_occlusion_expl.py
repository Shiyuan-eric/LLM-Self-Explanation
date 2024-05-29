from scipy import stats
import pandas as pd
import time
import random
import pickle
import ast
import os
import pickle
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
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
        "role": "user",
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



class OcclusionExplanationGenerator():
    def __init__(self, model, tokenizer, PE, max_tokens, dataset, device, noise, p_only, mistral, llama):
        # self.messages = messages
        self.tokenizer = tokenizer
        self.model = model
        self.max_tokens = max_tokens
        self.fails = 0
        self.total = 1
        # self.ovr_fails = 0
        # self.ovr_total = 1
        self.PE = PE
        self.dataset = dataset
        self.noise = noise
        self.device = device
        self.p_only = p_only
        self.mistral = mistral
        self.llama = llama

        # self.pre_phrase = "<review> "
        # self.post_phrase = " <review>"

        self.random_range = 10e-4
        random.seed(0)

    # Function to query the openai API and generate a gpt response given a prompt
    def generate_mistral_response(self, message):
        inputs = self.tokenizer.apply_chat_template(
            message, return_tensors="pt").to(device)
        outputs = self.model.generate(
            inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        last_instruction_idx = response.rindex("[/INST]") + 7
        return response[last_instruction_idx:]

    def generate_llama_response(self, message):
        input_ids = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            pad_token_id = self.tokenizer.eos_token_id,
            eos_token_id=terminators,
            # do_sample=True,
            # temperature=0.6,
            # top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return(self.tokenizer.decode(response, skip_special_tokens=True))

    def generate_response(self, message):
        if self.mistral:
            return self.generate_mistral_response(message)
        elif self.llama:
            return self.generate_llama_response(message)

    def parse_completion(self, response):
        lines = response.splitlines()
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
        model_response = self.generate_response(message)
        prediction, confidence, _ = self.parse_completion(model_response)
        if prediction == 1:
            score = confidence
        else:
            score = 1-confidence
        return score

    def generate_whole_prompt(self, sentence):
        if self.p_only:
            if self.mistral:
                prompt = mistral_P_MSG.copy()
            elif self.llama:
                prompt = llama_P_MSG.copy()
            # prompt = P_MSG.copy()
            prompt.append({"role": "user", "content": f"<review> {sentence} <review>"})
        elif self.PE:
            if self.mistral:
                prompt = mistral_P_E_MSG.copy()
            elif self.llama:
                prompt = llama_P_E_MSG.copy()
            # prompt = P_E_MSG.copy()
            prompt.append({"role": "user", "content": f"<review> {sentence} <review>"})
        else:
            if self.mistral:
                prompt = mistral_E_P_MSG.copy()
            elif self.llama:
                prompt = llama_E_P_MSG.copy()
            # prompt = E_P_MSG.copy()
            prompt.append({"role": "user", "content": f"<review> {sentence} <review>"})
        return prompt


    def compute_occlusion_saliency(self):
        # Model_Evaluator.get_completion()
        # if self.PE: # TODO: What's the new prompt for the open_source models?
        #     msg = P_E_MSG
        # else:
        #     msg = E_P_MSG
        if self.p_only:
            print("Using p only mode.")
        elif self.PE:
            print("Using PE mode.")
        else:
            print("Using EP mode.")

        occlusion_output = []
        sentence_list = []
        count = 0
        output = list()
        for sentence in tqdm(self.dataset, desc="sentence processing", position=0):
            full_msg = self.generate_whole_prompt(sentence)

            og_score = self.get_completion(full_msg)
            # print("The original score is:", og_score)
            sentences = sentence.split()
            occlusion = {}
            for i in tqdm(range(len(sentences)), desc="occlusion processing", position=1, leave=False):
                w = sentences.pop(i)
                ss = " ".join(sentences)
                full_new_msg = self.generate_whole_prompt(ss)
                new_result = self.get_completion(full_new_msg)
                # print("\nThe new score is:", new_result)
                if self.noise:
                    occlusion[i] = og_score - new_result + random.uniform(0, self.random_range)
                else:
                    occlusion[i] = og_score - new_result
                # print(f"The occlusion for {i}th value is {occlusion[i]}")
                sentences.insert(i, w)
            occlusion = dict(sorted(occlusion.items(), key=lambda item: item[1], reverse=True))
            sentence_list.append(sentence)
            # print(f"\n Occlusion = {occlusion}")
            occlusion_output.append(occlusion)
            attr = list()
            for k, v in occlusion.items():
                attr.append((sentences[k], (v, k)))
            print(attr)
            output.append((attr, sentences))
        
        if self.p_only:
            if self.llama:
                with open ('llama_P_only_Occlusion.pickle', 'wb') as dbfile:
                    pickle.dump(output, dbfile)
            elif self.mistral:
                with open ('mistral_P_only_Occlusion.pickle', 'wb') as dbfile:
                    pickle.dump(output, dbfile)
        elif self.PE:
            if self.llama:
                with open ('llama_PE_Occlusion.pickle', 'wb') as dbfile:
                    pickle.dump(output, dbfile)
            elif self.mistral:
                with open ('mistral_PE_Occlusion.pickle', 'wb') as dbfile:
                    pickle.dump(output, dbfile)
        else:
            if self.llama:
                with open ('llama_EP_Occlusion.pickle', 'wb') as dbfile:
                    pickle.dump(output, dbfile)
            elif self.mistral:
                with open ('mistral_EP_Occlusion.pickle', 'wb') as dbfile:
                    pickle.dump(output, dbfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mistral', action='store_true', help='the boolean value that indicates if the program will use the mistral model')
    parser.add_argument('-llama', action='store_true', help='the boolean value that indicates if the program will use the llama3 model')
    args = parser.parse_args()
    
    if args.mistral:
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    elif args.llama:
        MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset = load_dataset("sst")
    eval_ds = dataset["test"].shuffle(seed=8)
    
    print("\n *** dataset has been loaded *** \n")
    
    sentences = []
    labels = []
    count = 0
    NUM_ITER = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device is:', device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    max_tokens = 2048

    print(f"\n*** model {MODEL_NAME} has been loaded ***\n")

    # EP
    occlusion_generator = OcclusionExplanationGenerator(model=model, tokenizer=tokenizer, PE=False, max_tokens=max_tokens, dataset=eval_ds['sentence'][:NUM_ITER], device=device, noise=True, p_only=False, mistral=args.mistral, llama=args.llama)
    occlusion_generator.compute_occlusion_saliency()

    # PE
    occlusion_generator = OcclusionExplanationGenerator(model=model, tokenizer=tokenizer, PE=True, max_tokens=max_tokens, dataset=eval_ds['sentence'][:NUM_ITER], device=device, noise=True, p_only=False, mistral=args.mistral, llama=args.llama)
    occlusion_generator.compute_occlusion_saliency()

    # P_Only
    occlusion_generator = OcclusionExplanationGenerator(model=model, tokenizer=tokenizer, PE=False, max_tokens=max_tokens, dataset=eval_ds['sentence'][:NUM_ITER], device=device, noise=True, p_only=True, mistral=args.mistral, llama=args.llama)
    occlusion_generator.compute_occlusion_saliency()

    # occlusion_pickle = loadData('Open_Source_PE_Occlusion_Result_Predict')
    # for sentence, occlusion_value in occlusion_pickle.items():
    #     print(sentence, occlusion_value)


    