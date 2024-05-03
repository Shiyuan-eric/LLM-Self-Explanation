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


P_E_MSG = [
    {
        "role": "user",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
    ]

E_P_MSG = [
    {
        "role": "user",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
    ]

NUM_ITER = 100

class OcclusionExplanationGenerator():
    def __init__(self, model, tokenizer, PE, max_tokens, dataset, noise):
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

        # self.pre_phrase = "<review> "
        # self.post_phrase = " <review>"

        self.random_range = 10e-4
        random.seed(0)

    # Function to query the openai API and generate a gpt response given a prompt
    def generate_response(self, message):
        inputs = self.tokenizer.apply_chat_template(
            message, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        last_instruction_idx = response.rindex("[/INST]") + 7
        return response[last_instruction_idx:]

    def parse_completion(self, response):
        lines = response.splitlines()
        self.total += 1
        try:
            if self.PE:
                exp = ast.literal_eval(lines[1])
                (prediction, confidence) = ast.literal_eval(lines[0])
            else:
                exp = ast.literal_eval(lines[0])
                (prediction, confidence) = ast.literal_eval(lines[1])
        except:
            if not self.PE:
                try:
                    # Trying to see if the potential error was that there was a newline(something I saw a few times)
                    exp = ast.literal_eval(lines[0])
                    prediction, confidence = ast.literal_eval(lines[2])
                    return (prediction, confidence, exp)
                except:
                    pass
            # GPT didn't give an answer in the required format (more likely an invalid response)
            # So, make everything 0
            exp = []
            for token in response.split(' '):
                exp.append((token, 0.0))
            (prediction, confidence) = (0, 0.5)
            self.fails += 1
        return (prediction, confidence, exp)

    def get_completion(self, message):
        model_response = self.generate_response(message)
        return self.parse_completion(model_response)

    def generate_whole_prompt(self, sentence):
        if self.PE:
            prompt = P_E_MSG.copy()
            prompt.append({"role": "user", "content": f"<review> {sentence} <review>"})
        else:
            prompt = E_P_MSG.copy()
            prompt.append({"role": "user", "content": f"<review> {sentence} <review>"})
        return prompt


    def compute_occlusion_saliency(self):
        # Model_Evaluator.get_completion()
        # if self.PE: # TODO: What's the new prompt for the open_source models?
        #     msg = P_E_MSG
        # else:
        #     msg = E_P_MSG

        occlusion_output = []
        sentence_list = []
        count = 0
        for sentence in tqdm(self.dataset, desc="sentence processing", position=0):
            full_msg = self.generate_whole_prompt(sentence)
            if self.PE: # TODO: What's the new prompt for the open_source models?
                og_score, _, _ = self.get_completion(full_msg) # TODO: What's the range of the prediction?
            else:
                og_score, _, _ = self.get_completion(full_msg) # TODO: What's the range of the prediction?
            sentences = sentence.split()
            occlusion = {}
            for i in tqdm(range(len(sentences)), desc="occlusion processing", position=1, leave=False):
                w = sentences.pop(i)
                ss = " ".join(sentences)
                full_new_msg = self.generate_whole_prompt(ss)
                new_result, _, _ = self.get_completion(full_new_msg) # TODO: What's the range of the prediction?
                if self.noise:
                    occlusion[i] = og_score - new_result + random.uniform(0, self.random_range)
                else:
                    occlusion[i] = og_score - new_result
                sentences.insert(i, w)
            occlusion = dict(sorted(occlusion.items(), key=lambda item: item[1], reverse=True))
            sentence_list.append(sentence)
            # print(f"\n Occlusion = {occlusion}")
            occlusion_output.append(occlusion)
        if self.PE:
            with open ('Open_Source_PE_Occlusion_Result_Predict', 'wb') as dbfile:
                pickle.dump(dict(zip(sentence_list,occlusion_output)), dbfile)
        else:
            with open ('Open_Source_EP_Occlusion_Result_Predict', 'wb') as dbfile:
                pickle.dump(dict(zip(sentence_list,occlusion_output)), dbfile)        

def loadData(filename):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


if __name__ == "__main__":
    dataset = load_dataset("sst")
    eval_ds = dataset["test"].shuffle(seed=8)
    
    print("\n *** dataset has been loaded *** \n")
    
    sentences = []
    labels = []
    count = 0
    num_examples = 100

    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    max_tokens = 2048

    print(f"\n*** model {MODEL_NAME} has been loaded ***\n")

    occlusion_generator = OcclusionExplanationGenerator(model, tokenizer, False, max_tokens, eval_ds['sentence'][:NUM_ITER], noise=True)
    occlusion_generator.compute_occlusion_saliency()

    # occlusion_pickle = loadData('Open_Source_PE_Occlusion_Result_Predict')
    # for sentence, occlusion_value in occlusion_pickle.items():
    #     print(sentence, occlusion_value)


    