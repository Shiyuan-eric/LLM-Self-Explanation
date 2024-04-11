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

PRE_PHRASE = "<review> "
POST_PHRASE = " <review>"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16).to(device)


dataset = load_dataset("sst")

eval_ds = dataset["test"].shuffle(seed=8)
dataloader = DataLoader(eval_ds, batch_size=1)

print("loaded dataset and device")


sentences = []
labels = []
count = 0
num_examples = 100
for batch in dataloader:
    if count == num_examples:
        break
    sentences.append(batch['sentence'][0])
    labels.append(batch['label'].item())
    count += 1

messages = []
responses = {}


def generate_response():
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    last_instruction_idx = response.rindex("[/INST]") + 7
    return response[last_instruction_idx:]


def topk_prompt(k):
    prompt = "As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word and punctuation in determining the overall positivity level. Your task is to identify the top %d most significant words, ranked from the most positive sentiment to the least positive sentiment. Additionally, you need to determine whether the movie review is positive or negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nThe movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review's content.\n\nOnly output the list of %d words in the form of a comma separated list with spaces, and the prediction(as a number) and confidence following after, nothing more. Do you understand?" % (
        k, k)
    return prompt


def parse_topk(response, k, sentence):
    connected = ""
    for elem in response.splitlines():
        connected += elem.strip("\n")
    wsf = 0  # words so far
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

        if (not isinstance(number_list[0], int) or not isinstance(number_list[1], float) or wsf != k):
            raise Exception
        for word in word_list:
            if word not in connected.split(" "):
                raise Exception
        output = (word_list, number_list)
    except Exception:
        print("Failed. Response: " + response +
              " for input " + sentence + ". K was " + str(k))
        print(
            "Enter the correct parsing in the form ([word, word, word, ...], [pred, conf])")
        correct = input()
        output = eval(correct)
    return output


for i in tqdm(range(num_examples)):
    k = int(len(sentences[i].split(" ")) * 0.2)
    if k < 1:
        k = 1
    messages.append({"role": "user", "content": topk_prompt(k)})
    messages.append(
        {"role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."})
    messages.append({"role": "user", "content": PRE_PHRASE +
                    sentences[i] + POST_PHRASE})
    completion = generate_response()
    responses[sentences[i]] = parse_topk(
        completion, k, sentences[i])
    messages.pop()
    messages.pop()
    messages.pop()

# print(responses)
with open("parse_topk.pickle", "wb") as handle:
    pickle.dump(responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("labels_PE.pickle", "wb") as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
