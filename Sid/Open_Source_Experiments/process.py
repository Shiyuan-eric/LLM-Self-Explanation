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
import argparse

from model_evaluator import Model_Evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The program process.py takes an explanation input, and calulates corresponding Comprehensiveness, Sufficiency, Df Mit, Df Frac, and Deletion Rank Csorrelation.')
    parser.add_argument('-m', '--max_token', type=int, default=2048, help='input the max_token argument')
    parser.add_argument('-r', '--response_file_name', type=str, help='name the input the file which stores explanations response', required=True)
    parser.add_argument('-o', '--output_file', type=str, help='name the output file ', required = True)
    parser.add_argument('-pe', action='store_true', help='the boolean value that indicates whether the process is using pe(Predict and Explain)[True] or ep(Explain then Predict)[False]')
    parser.add_argument('-sh', action='store_true', help='an indicater of whether the input response file was in Shiyuan\'s Format')
    args = parser.parse_args()

    print(args.pe)
    
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(device)

    
    label_filename = "labels_PE.pickle"

    output_file = open(args.output_file, 'w')

    evaluator = Model_Evaluator(model=model, tokenizer=tokenizer, response_filename=args.response_file_name,
                                PE=args.pe, label_filename=label_filename, max_tokens=args.max_token)
    
    print("Input File: " + response_filename, file=output_file)
    if args.sh:
        evaluator.reconstruct_expl()
    else:
        evaluator.process_model_input()
        evaluator.print_fail_rate()
        evaluator.reset_fails()
    
    print(evaluator.explanations[0])
    
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
