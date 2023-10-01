import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import getopt, sys
import time

# Prediction Only
# MESSAGES = [
#     {
#       "role": "system",
#       "content": "You are a creative   and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nExample output:\n(<int classification>, <float confidence>)"
#     }
#   ]
# PE
# MESSAGES = [
#     {
#       "role": "system",
#       "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
#     }
#   ]
#EP
MESSAGES = [
    {
      "role": "system",
      "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
    }
  ]

PRE_PHRASE = "<review> "
POST_PHRASE = " <review>"


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
num_examples = 100
for batch in dataloader:
#  print(batch)
 if count == num_examples:
  break
 sentences.append(batch['sentence'][0])
 labels.append(batch['label'].item())
 count+=1
print("starting api calls")

openai.api_key = "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"
messages = MESSAGES[:]
responses = {}

def generate_response(model=MODEL):
  while True:
    try:
        response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        break
    except openai.error.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
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
  return response

for i in tqdm(range(num_examples)):
  messages.append({"role": "user", "content": PRE_PHRASE + sentences[i] + POST_PHRASE})
  completion = generate_response(model=MODEL)
  responses[sentences[i]] = completion.choices[0].message.content
  messages.pop()
  # messages.append({"role": "assistant", "content": completion.choices[0].message.content})
  

# print(responses)
with open("gpt_response_EP.pickle", "wb") as handle:
  pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("labels_EP.pickle", "wb") as handle:
   pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)