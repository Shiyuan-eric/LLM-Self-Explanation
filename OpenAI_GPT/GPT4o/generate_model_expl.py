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
import argparse
from openai import OpenAI
import argparse



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
PRE_PHRASE = "<review> "
POST_PHRASE = " <review>"
MODEL="gpt-4o"
TEMPERATURE = 0
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_response(messages):
    
    while True:
        try:
            response = client.chat.completions.create(model=MODEL,
                                                      messages=messages,
                                                      temperature=0,
                                                      max_tokens=1024,
                                                      top_p=1,
                                                      frequency_penalty=0,
                                                      presence_penalty=0)
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
    return response.choices[0].message.content

def loadData(filename):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
def storeData(filename, data):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

sentences = loadData('sentences.pickle')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-pe', action='store_true', help='')
    args = parser.parse_args()
    responses = {}
    for i in tqdm(range(4)):
        if args.pe:
            messages = gpt4o_PE_MSG.copy()
        else:
            messages = gpt4o_EP_MSG.copy()

        messages.append({"role": "user", "content": [{"type": "text", "text": 
                PRE_PHRASE + sentences[i] + POST_PHRASE}]})
        completion = generate_response(messages)
        responses[sentences[i]] = completion
        print(completion)
    # if args.pe:
    #     storeData("gpt_response_PE.pickle", responses)
    #     # storeData("labels_PE.pickle", labels)
    # else:
    #     storeData("gpt_response_EP.pickle", responses)