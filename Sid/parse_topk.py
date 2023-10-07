import re
import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import getopt, sys
import time

MESSAGES = [
    {
      "role": "system",
      "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
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
 if count == num_examples:
  break
 sentences.append(batch['sentence'][0])
 labels.append(batch['label'].item())
 count+=1
print("starting api calls")

openai.api_key = "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"
messages = []
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

def topk_prompt(k):
   prompt = "As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word and punctuation in determining the overall positivity level. Your task is to identify the top %d most significant words, ranked from the most positive sentiment to the least positive sentiment. Additionally, you need to determine whether the movie review is positive or negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nThe movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review's content.\n\nOnly output the list of %d words in the form of a comma separated list with spaces, and the prediction(as a number) and confidence following after, nothing more." % (k, k)
   return prompt

def parse_topk(response, k, sentence):
    connected = ""
    for elem in response.splitlines():
        connected += elem.strip("\n")
    wsf = 0 #words so far
    tkns_list = connected.split(" ")
    word_list = []
    number_list = []
    other_words = []
    try:
        for word in tkns_list:
            if wsf < k and word[-1] == "," and not bool(re.search(r'\d', word[:-1])):
                word_list.append(word[:-1])
                wsf += 1
            elif (word[-1] == "," and word[:-1].replace(".", "").isnumeric()) or word.replace(".", "").isnumeric():
                number_list.append(eval(word[:-1]))
            else:
                other_words.append(word)
        
        if(not isinstance(number_list[0], int) or not isinstance(number_list[1], float) or wsf != k):
           raise Exception
        for word in word_list:
           if word not in connected.split(" "):
              raise Exception
        output = (word_list, number_list)
    except Exception:
       print("Failed. Response: " + response + " for input " + sentence + ". K was " + str(k))
       print("Enter the correct parsing in the form ([word, word, word, ...], [pred, conf])")
       correct = input()
       output = eval(correct)
    return output

for i in tqdm(range(num_examples)):
  k = int(len(sentences[i].split(" ")) * 0.2)
  if k < 1:
     k = 1
  messages.append({"role": "system", "content": topk_prompt(k)})
  messages.append({"role": "user", "content": PRE_PHRASE + sentences[i] + POST_PHRASE})
  completion = generate_response(model=MODEL)
  responses[sentences[i]] = parse_topk(completion.choices[0].message.content, k, sentences[i])
  messages.pop()
  messages.pop()
  
# print(responses)
with open("parse_topk.pickle", "wb") as handle:
  pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("labels_PE.pickle", "wb") as handle:
   pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


