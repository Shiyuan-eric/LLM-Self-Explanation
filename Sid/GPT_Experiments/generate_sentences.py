# Simple script to pick random 100 sentences from sst dataset and save them in a pickle file

from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

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

with open("sentences.pickle", "wb") as handle:
    pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
