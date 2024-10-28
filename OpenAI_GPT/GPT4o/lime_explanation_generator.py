# LIME
import os
from tqdm import tqdm
import numpy as np
import lime
from lime.lime_text import IndexedString
from lime.lime_text import LimeTextExplainer
from lime import lime_text
import random
import ast
import pickle
import re

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

def loadData(filename):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
def storeData(filename, data):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

class LimeExplanationGenerator():
    # Init
    def __init__(self, response_filename, PE, start, end):

        self.messages = ""
        with open(response_filename, 'rb') as handle:
            self.sentences = pickle.load(handle)
        self.sentences = self.sentences[start:end]

        self.fails = 0
        self.total = 1
        self.ovr_fails = 0
        self.ovr_total = 1
        self.response_filename = response_filename
        self.PE = PE
        if self.PE:
            self.cache_name = 'gpt4o_pe_cache.pickle'
        else:
            self.cache_name = 'gpt4o_ep_cache.pickle'

        self.pre_phrase = "<review> "
        self.post_phrase = " <review>"

        self.random_range = 10e-4
        random.seed(0)

    def init_message(self):
        if self.PE:
            self.messages = gpt4o_PE_MSG.copy()
        else:
            self.messages = gpt4o_EP_MSG.copy()

    def generate_response(self):
        return generate_response(self.messages)

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
    
    def get_result(self, phrase):
        cache_dict = loadData(self.cache_name)
        if phrase in cache_dict:
            label, prob = cache_dict[phrase]
        else:
            self.init_message()
            self.messages.append({"role": "user", "content": [{"type": "text", "text": 
                self.pre_phrase + phrase + self.post_phrase}]})
            label, prob, _ = self.parse_completion(self.generate_response())
            self.messages.pop()
            cache_dict[phrase] = (label, prob)
            storeData(self.cache_name, cache_dict)
        return label, prob

    # predict_proba function returning GPT confidence for LimeTextExplainer
    def predict_proba(self, sentences):
        probs = np.zeros((len(sentences), 2), dtype=float)
        for i in range(len(sentences)):
            phrase = sentences[i]
            (pred, conf) = self.get_result(phrase)
            # self.messages.append(
                # {"role": "user", "content": self.pre_phrase + sentences[i] + self.post_phrase})
            # (pred, conf, _) = self.parse_completion(
                # self.generate_response())
            # self.messages.pop()
            # print(i, pred)
            try:
                if pred > 1:
                    pred = 1
                elif pred < 0:
                    pred = 0
                probs[i, pred] = conf
                probs[i, 1-pred] = 1 - conf
            except:
                probs[i, 1] = 0.5
                probs[i, 0] = 0.5
        return probs

    # Uses LimeTextExplainer to compute LIME saliency
    def compute_lime_saliency(self, class_names):
        self.explanations = []
        explainer = LimeTextExplainer(class_names=class_names, bow=False)
        for sentence in tqdm(self.sentences):
            # get LIME explanations
            sent = sentence
            orig_tokens = sentence.split(' ')
            indexed_string = IndexedString(sent, bow=False)
            exp = explainer.explain_instance(sent, self.predict_proba, num_features=indexed_string.num_words(
            ), num_samples=(10 * indexed_string.num_words()))
            exp = exp.as_list()

            lime_tkns = []
            new_exp = []

            for i in range(len(exp)):
                lime_tkns.append(exp[i][0])

            # match saliency values to original tokens
            for i in range(len(orig_tokens)):
                try:
                    idx = lime_tkns.index(orig_tokens[i])
                except:
                    idx = -1
                if idx != -1:
                    new_exp.append((lime_tkns[idx], (exp[idx][1], i)))
                    lime_tkns[idx] = ''
                else:  # Random small value (0 but with uniqueness)
                    new_exp.append(
                        (orig_tokens[i], (random.uniform(-1 * self.random_range, self.random_range), i)))

            new_exp = sorted(new_exp, key=lambda x: x[1][0], reverse=True)
            self.explanations.append((new_exp, orig_tokens))
