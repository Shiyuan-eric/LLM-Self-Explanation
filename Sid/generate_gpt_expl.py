import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import getopt, sys

# Model parameters
# MESSAGES = [
#     {"role": "system", "content": "You are a movie review bot, whose job is to aid in sentiment analysis of movie reviews."},
#     {"role": "user", "content": "You are a movie review bot. Your job is to analyze movie reviews and classify them as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. Please also provide the degree of importance of each word in the sentence in your explanation in the format (\"word\", importance). For example, the expected response for the sentence \"This movie was very good\" is: \"Classification: 1, 0.99 confidence. [(\"This\", 0.0), (\"movie\", 0.1), (\"was\", -0.1), (\"very\", 0.6), (\"good\", 0.9)].  It does not matter whether or not the sentence makes sense. Do your best given the sentence.\""},
#     {"role": "assistant", "content": "I understand."},
#     {"role": "user", "content": "Now the next sentence: This movie was very bad"},
#     {"role": "assistant", "content": "Classification: 0, 0.99 confidence. [(\"This\", 0.096), (\"movie\", 0.194), (\"was\", -0.124), (\"very\", 0.606), (\"bad\", -0.918)]"},
#     {"role": "user", "content": "Now the next sentence: There is not a single movie that could have been better than this."},
#     {"role": "assistant", "content": "Classification: 1, 0.90 confidence. [(\"There\", 0.004), (\"is\", 0.114), (\"not\", -0.787), (\"a\", 0.119), (\"single\", 0.239, (\"movie\", 0.395, (\"that\", 0.043), (\"could\", 0.294), (\"have\", 0.155), (\"been\", 0.020), (\"better\", 0.859), (\"than\", 0.122), (\"this\", 0.500)]"},
#     {"role": "user", "content": "Now the next sentence: It was a great movie overall, but the ending was a bit lackluster."},
#     {"role": "assistant", "content": "Classification: 1, 0.75 confidence. [(\"It\", 0.174), (\"was\", -0.101), (\"a\", 0.122), (\"great\", 0.825), (\"movie\", 0.608), (\"overall\", 0.390), (\"but\", -0.134), (\"the\", 0.033), (\"ending\", -0.635), (\"was\", -0.145), (\"a\", 0.103), (\"bit\", -0.396), (\"lackluster\", -0.859)]"}
#   ]
# Testing only PE for now
MESSAGES = [
    {
      "role": "system",
      "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. \n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    }
  ]
TEMPERATURE = 0
MODEL = "gpt-3.5-turbo"

from datasets import load_dataset

dataset = load_dataset("sst")

eval_ds = dataset["test"].shuffle(seed=8)
dataloader = DataLoader(eval_ds, batch_size=1)

print("loaded dataset and device")

import openai

sentences = []
labels = []
count = 0
for batch in dataloader:
 if count == 100:
  break
 sentences.append(batch['sentence'][0])
 labels.append(batch['label'].item())
 count+=1
# print(sentences)
# print(labels)
print("starting api calls")

openai.api_key = "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"
messages = MESSAGES[:]
responses = []
num_examples = 3
for i in tqdm(range(num_examples)):
  
  messages.append({"role": "user", "content": sentences[i]})
  completion = openai.ChatCompletion.create(
    model=MODEL,
    messages=messages,
    temperature=TEMPERATURE
  )
  responses.append(completion.choices[0].message.content)
  messages.pop()
  # messages.append({"role": "assistant", "content": completion.choices[0].message.content})
  

# print(responses)
with open("gpt_response.pickle", "wb") as handle:
  pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("sentences.pickle", "wb") as handle:
  pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)