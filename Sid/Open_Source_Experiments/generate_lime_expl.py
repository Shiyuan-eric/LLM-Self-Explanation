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


# PE
messages = [
    {
        "role": "user", "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence. Ensure your output is exactly as described in the format, with nothing more or less. \n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
]


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16).to(device)

max_tokens = 1024

print("loaded model")


start = 0
end = 1
batch_size = 1  # checkpoints every batch_size batches into a pickle file
for i in tqdm(range(start, end, batch_size)):
    lime_generator = LimeExplanationGenerator(
        response_filename='sentences.pickle', model=model, tokenizer=tokenizer, max_tokens=max_tokens, PE=True, messages=messages, start=i, end=i+batch_size)
    lime_generator.compute_lime_saliency([0, 1])
    with open("LIME_response_PE_%d_%d.pickle" % (i, i+batch_size), "wb") as handle:
        pickle.dump(lime_generator.explanations, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
prefixed = [filename for filename in os.listdir(
    '.') if filename.startswith("LIME_response_PE_")]
explanations = []
for filename in prefixed:
    with open(filename, 'rb') as handle:
        cur = pickle.load(handle)
    for e in cur:
        explanations.append(e)
with open("LIME_response_PE_%d_%d.pickle" % (start, end), "wb") as handle:
    pickle.dump(explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)
