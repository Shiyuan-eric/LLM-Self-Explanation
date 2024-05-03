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

# PE (PredictThenExplain prompt)
MESSAGES = [
    {
        "role": "user", "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence. Ensure your output is exactly as described in the format, with nothing more or less. \n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
]

dataset = load_dataset("sst")

eval_ds = dataset["test"].shuffle(seed=8)
dataloader = DataLoader(eval_ds, batch_size=1)
print("loaded dataset")

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

print(sentences)

PRE_PHRASE = "<review> "
POST_PHRASE = " <review>"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16).to(device)

print("loaded model")


messages = MESSAGES[:]
responses = {}

# generates response


def generate_response():
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    last_instruction_idx = response.rindex("[/INST]") + 7
    return response[last_instruction_idx:]


# get completion for each sentence
for i in tqdm(range(num_examples)):
    messages.append({"role": "user", "content": PRE_PHRASE +
                    sentences[i] + POST_PHRASE})
    completion = generate_response()
    responses[sentences[i]] = completion
    messages.pop()

# Make sure to change file names if changing from PE
with open("mistral7B_response_PE.pickle", "wb") as handle:
    pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("labels_PE.pickle", "wb") as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("sentences.pickle", "wb") as handle:
    pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
