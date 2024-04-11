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


class LimeExplanationGenerator():
    # Init
    def __init__(self, response_filename, model, tokenizer, max_tokens, PE, messages, start=0, end=100):

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

        self.pre_phrase = "<review> "
        self.post_phrase = " <review>"

        self.random_range = 10e-4
        random.seed(0)

    def generate_response(self):
        inputs = self.tokenizer.apply_chat_template(
            self.messages, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        last_instruction_idx = response.rindex("[/INST]") + 7
        return response[last_instruction_idx:]

    def parse_completion(self, response):
        # print(response)
        lines = response.splitlines()
        try:
            if self.PE:
                exp = ast.literal_eval(lines[1])
                (prediction, confidence) = ast.literal_eval(lines[0])
            else:
                exp = ast.literal_eval(lines[0])
                (prediction, confidence) = ast.literal_eval(lines[1])
        except:
            exp = []
            for token in response.split(' '):
                exp.append((token, 0.0))
            (prediction, confidence) = (0, 0.5)
            self.fails += 1
        self.total += 1
        return (prediction, confidence, exp)

    # predict_proba function returning GPT confidence for LimeTextExplainer
    def predict_proba(self, sentences):
        probs = np.zeros((len(sentences), 2), dtype=float)
        for i in range(len(sentences)):
            self.messages.append(
                {"role": "user", "content": self.pre_phrase + sentences[i] + self.post_phrase})
            (pred, conf, expl) = self.parse_completion(
                self.generate_response())
            self.messages.pop()
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
