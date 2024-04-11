from scipy import stats
import pandas as pd
import time
import random
import pickle
import ast
import os
import pickle
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_evaluator import Model_Evaluator


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_TOKENS = 1024


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16).to(device)


response_filename = "mistral7B_response_PE.pickle"
label_filename = "labels_PE.pickle"
output_filename = "mistral_process.txt"
output_file = open(output_filename, 'w')

messages = [
    {
        "role": "user", "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence. Ensure your output is exactly as described in the format, with nothing more or less. \n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
]
evaluator = Model_Evaluator(model=model, tokenizer=tokenizer, response_filename=response_filename,
                            PE=True, messages=messages, label_filename=label_filename, max_tokens=MAX_TOKENS)

print("Input File: " + response_filename, file=output_file)
evaluator.process_model_input()
evaluator.print_fail_rate()
evaluator.reset_fails()

accuracy = evaluator.calculate_accuracy()
print("Accuracy: ", str(accuracy), file=output_file)

mistral_comprehensiveness = evaluator.calculate_comprehensiveness()
print("Mistral Comprehensiveness: ", str(
    mistral_comprehensiveness), file=output_file)
evaluator.print_fail_rate()
evaluator.reset_fails()

mistral_sufficiency = evaluator.calculate_sufficiency()
print("Mistral Sufficiency: ", str(mistral_sufficiency), file=output_file)
evaluator.print_fail_rate()
evaluator.reset_fails()

mistral_df_mit = evaluator.calculate_DF_MIT()
print("Mistral DF_MIT: ", str(mistral_df_mit), file=output_file)
evaluator.print_fail_rate()
evaluator.reset_fails()

mistral_df_frac = evaluator.calculate_DF_Frac()
print("Mistral DF_Frac: ", str(mistral_df_frac), file=output_file)
evaluator.print_fail_rate()
evaluator.reset_fails()

mistral_del_rank_correlation = evaluator.calculate_del_rank_correlation()
print("Mistral Deletion Rank Correlation: ", str(
    mistral_del_rank_correlation), file=output_file)
evaluator.print_fail_rate()
evaluator.reset_fails()

metric_values = [mistral_comprehensiveness, mistral_sufficiency,
                 mistral_df_mit, mistral_df_frac, mistral_del_rank_correlation]
metric_names = ["Comprehensivness", "Sufficiency",
                "DF_MIT", "DF_Frac", "Deletion Rank Correlation"]
metric_df = pd.DataFrame(metric_values, metric_names)
print(metric_df, file=output_file)
print("\nComplete!", file=output_file)
