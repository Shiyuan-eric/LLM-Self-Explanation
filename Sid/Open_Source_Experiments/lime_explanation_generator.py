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

class LimeExplanationGenerator():
    # Init
    def __init__(self, response_filename, model, tokenizer, max_tokens, PE, messages, start, end, mistral, llama):

        self.messages = messages
        with open(response_filename, 'rb') as handle:
            self.sentences = pickle.load(handle)
        self.sentences = self.sentences[start:end]

        self.tokenizer = tokenizer
        self.model = model
        self.max_tokens = max_tokens
        self.fails = 0
        self.total = 1
        self.ovr_fails = 0
        self.ovr_total = 1
        self.response_filename = response_filename
        self.PE = PE
        self.mistral = mistral
        self.llama = llama

        self.pre_phrase = "<review> "
        self.post_phrase = " <review>"

        self.random_range = 10e-4
        random.seed(0)

    def generate_mistral_response(self):
        inputs = self.tokenizer.apply_chat_template(
            self.messages, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        last_instruction_idx = response.rindex("[/INST]") + 7
        return response[last_instruction_idx:]
    
    def generate_llama_response(self):
        input_ids = self.tokenizer.apply_chat_template(
            self.messages,
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

    def generate_response(self):
        if self.mistral:
            return self.generate_mistral_response()
        elif self.llama:
            return self.generate_llama_response()

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

    # predict_proba function returning GPT confidence for LimeTextExplainer
    def predict_proba(self, sentences):
        probs = np.zeros((len(sentences), 2), dtype=float)
        for i in range(len(sentences)):
            self.messages.append(
                {"role": "user", "content": self.pre_phrase + sentences[i] + self.post_phrase})
            (pred, conf, _) = self.parse_completion(
                self.generate_response())
            self.messages.pop()
            # print(i, pred)
            if pred > 1:
                pred = 1
            elif pred < 0:
                pred = 0
            probs[i, pred] = conf
            probs[i, 1-pred] = 1 - conf
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
