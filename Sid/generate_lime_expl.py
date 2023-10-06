# LIME
import lime
import pickle
import openai
import ast
import time
import random
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from lime.lime_text import IndexedString
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import retry
import os
import time

openai.api_key = "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"


TEMPERATURE = 0
MODEL = "gpt-3.5-turbo"
PRE_PHRASE = "<review> "
POST_PHRASE = " <review>"

random_range = 10e-4
random.seed(0)


class LimeExplanationGenerator():
  
    def __init__(self, filename, PE, messages, start=0, end=100):

        self.messages = messages
        self.total = 0
        self.fails = 0
        with open(filename, 'rb') as handle:
            self.sentences = pickle.load(handle)
        self.sentences = self.sentences[start:end]
        self.PE = PE

    def generate_response(self):
        while True:
            try:
                response = openai.ChatCompletion.create(
                model=MODEL,
                messages=self.messages,
                temperature=TEMPERATURE,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
                break
            except openai.error.RateLimitError as e:
                retry_time = e.retry_after if hasattr(e, 'retry_after') else 60
                print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                continue
            except openai.error.Timeout as e:
                print(f"Request timed out: {e}. Retrying in 10 seconds...")
                time.sleep(10)
                continue
            except openai.error.APIError as e:
                retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
                print(f"API error occurred. Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                continue
            except openai.error.ServiceUnavailableError as e:
                print(f"Service is unavailable. Retrying in 10 seconds...")
                time.sleep(10)
                continue
            except openai.error.APIConnectionError as e:
                print(f"Not connected to internet. Retrying in 300 seconds...")
                time.sleep(300)
                continue
        return response

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
            # GPT didn't understand/format strayed. Go in the middle.
            exp = []
            for token in response.split(' '):
                exp.append((token, 0.0))
            (prediction, confidence) = (0, 0.5)
            self.fails += 1
        self.total += 1
        return (prediction, confidence, exp)
    
    
    def predict_proba(self, sentences):
        probs = np.zeros((len(sentences), 2), dtype=float)
        for i in range(len(sentences)):
            messages.append({"role": "user", "content": PRE_PHRASE + sentences[i] + POST_PHRASE})
            (pred, conf, expl) = self.parse_completion(self.generate_response().choices[0].message.content)
            messages.pop()
            probs[i, pred] = conf
            probs[i, 1-pred] = 1 - conf
        return probs

    def compute_lime_saliency(self, class_names):
        self.explanations = []
        explainer = LimeTextExplainer(class_names=class_names, bow=False)
        for sentence in tqdm(self.sentences):
            
            sent = sentence
            orig_tokens = sentence.split(' ')
            indexed_string = IndexedString(sent, bow=False)
            exp = explainer.explain_instance(sent, self.predict_proba, num_features=indexed_string.num_words(), num_samples=(10 * indexed_string.num_words()))
            exp = exp.as_list()

            lime_tkns = []
            new_exp = []

            for i in range(len(exp)):
                lime_tkns.append(exp[i][0])
            
            for i in range(len(orig_tokens)):
                try:
                    idx = lime_tkns.index(orig_tokens[i])
                except:
                    idx = -1
                if idx != -1:
                    new_exp.append((lime_tkns[idx], (exp[idx][1], i)))
                    lime_tkns[idx] = ''
                else:
                    new_exp.append((orig_tokens[i], (random.uniform(-1 * random_range, random_range), i)))
            
            new_exp = sorted(new_exp, key=lambda x: x[1][0], reverse=True)
            self.explanations.append((new_exp, orig_tokens))

        
    

if __name__ == "__main__":
    #EP
    messages = [
        {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
        }
    ]
    start = 50
    end = 100
    batch_size = 5
    for i in range(start, end, batch_size):
        lime_generator = LimeExplanationGenerator('sentences.pickle', PE=False, messages=messages, start=i, end=i+batch_size)
        lime_generator.compute_lime_saliency([0, 1])
        with open("LIME_response_EP_%d_%d.pickle" % (i, i+batch_size), "wb") as handle:
            pickle.dump(lime_generator.explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    prefixed = [filename for filename in os.listdir('.') if filename.startswith("LIME_response_EP_")]
    explanations = []
    for filename in prefixed:
        with open(filename, 'rb') as handle:
            cur = pickle.load(handle)
        for e in cur:
            explanations.append(e)
    with open("LIME_response_EP_%d_%d.pickle" % (start, end), "wb") as handle:
        pickle.dump(explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    