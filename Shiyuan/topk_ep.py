import re
import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import getopt, sys
import time
import os



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

openai.api_key = os.getenv("OPENAI_API_KEY")
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
      print(f"\nRate limit exceeded. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      continue
    except openai.error.Timeout as e:
      print(f"\nRequest timed out: {e}. Retrying in 10 seconds...")
      time.sleep(10)
      continue
    except openai.error.APIError as e:
      retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
      print(f"\nAPI error occurred. Retrying in {retry_time} seconds...")
      time.sleep(retry_time)
      continue
    except openai.error.ServiceUnavailableError as e:
      print(f"\nService is unavailable. Retrying in 10 seconds...")
      time.sleep(10)
      continue
    except OSError as e:
      if isinstance(e, tuple) and len(e) == 2 and isinstance(e[1], OSError):
          retry_time = 10  # Adjust the retry time as needed
          print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
          time.sleep(retry_time)
          continue
      else:
          retry_time = 10  # Adjust the retry time as needed
          print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
          time.sleep(retry_time)
          continue
  return response

def topk_prompt(k):
   prompt = "As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word and punctuation in determining the overall positivity level. Your task is to identify the top %d most significant words, ranked from the most positive sentiment to the least positive sentiment. Additionally, you need to determine whether the movie review is positive or negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Please note that the coherence of the sentence is not relevant; your focus should be on analyzing the sentiment.\n\nThe movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review's content.\n\nOnly output the list of %d words in the form of a comma separated list, with the prediction(as a number) and confidence following after, nothing more." %(k, k)
   return prompt

def parse_topk(response, k, sentence):
  print("response:")
  print(response)
  print("*************")
  print("k:", k)
  print("*************")
  print("sentence:")
  print(sentence)
  
  wordlist = []
  for i in range(k):
    while True:
      word = input(f"Please enter the {i}th most important word: ")
      if word in sentence.split():
        wordlist.append(word)
        break
      else:
        continue
  while True:
    try:
        pred = float(input(f"Prediction: "))
        conf = float(input(f"Confidence: "))
        break
    except:
        continue
  return (wordlist, [pred, conf])
  
    # connected = ""
    # for elem in response.splitlines():
    #     connected += elem.strip("\n")
    # wsf = 0
    # tkns_list = connected.split(" ")
    # word_list = []
    # number_list = []
    # other_words = []
    # try:
    #     print(tkns_list)
    #     for word in tkns_list:
    #         word.strip(",")
    #         if wsf < k and word[-1] == "," and not bool(re.search(r'\d', word[:-1])):
    #             word_list.append(word)
    #             wsf += 1
    #         elif (word[-1] == "," and word[:-1].replace(".", "").isnumeric()) or word.replace(".", "").isnumeric():
    #             number_list.append(eval(word[:-1]))
    #         else:
    #             other_words.append(word)
    #     print(wsf, word_list)
    #     if(not isinstance(number_list[0], int) or not isinstance(number_list[1], float) or wsf != k):
    #        raise Exception
    #     for word in word_list:
    #        if word not in connected.split(" "):
    #           raise Exception
    #     output = (word_list, number_list)
    # except Exception:
    #    print("Failed. Response: " + response + " for input " + sentence + ". K was " + str(k))
    #    print("Enter the correct parsing in the form ([word, word, word, ...], [pred, conf])")
    #    correct = input()
    #    output = eval(correct)
    # return output

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
with open("parse_topk_ep.pickle", "wb") as handle:
  pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("labels_PE.pickle", "wb") as handle:
   pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


