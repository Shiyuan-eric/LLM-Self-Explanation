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
import sys

from model_evaluator import Model_Evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The program process.py takes an explanation input, and calulates corresponding Comprehensiveness, Sufficiency, Df Mit, Df Frac, and Deletion Rank Correlation.')
    parser.add_argument('-m', '--max_token', type=int, default=2048, help='input the max_token argument')
    parser.add_argument('-r', '--response_file_name', type=str, help='name the input the file which stores explanations response', required=True)
    parser.add_argument('-l', '--label_file_name', type=str, help='name the input the file which stores all the predicted labels')
    parser.add_argument('-o', '--output_file', type=str, help='name the output file ')
    parser.add_argument('-pe', action='store_true', help='the boolean value that indicates whether the process is using pe(Predict and Explain)[True] or ep(Explain then Predict)[False]')
    parser.add_argument('-p_only', action='store_true', help='the boolean value that indicates whether the process will be using p_only mode')
    parser.add_argument('-topk', action='store_true', help='the boolean value that indicates whether the explanation is generated in topk format')
    # parser.add_argument('-ref', '--reference_file', type=str, help='name the file for reference (usually for PE and EP).'
    parser.add_argument('-occ', action='store_true', help='an indicater of whether the input response file was in Shiyuan\'s Format')
    parser.add_argument('-lime', action='store_true', help='an indicater of whether the input response file was from LIME')
    parser.add_argument('-mistral', action='store_true', help='the boolean value that indicates if the program will use the mistral model')
    parser.add_argument('-llama', action='store_true', help='the boolean value that indicates if the program will use the llama3 model')
    args = parser.parse_args()
    
    if args.mistral:
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    elif args.llama:
        MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device = {device}, PE = {args.pe}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(device)

    label_filename = "true_labels.pickle"
    if args.output_file:
        output_file = open(args.output_file, 'w')
    else:
        output_file = sys.stdout
    evaluator = Model_Evaluator(model=model, tokenizer=tokenizer, response_filename=args.response_file_name,
                                PE=args.pe, label_filename=label_filename, max_tokens=args.max_token, p_only=args.p_only,
                                mistral=args.mistral, llama=args.llama)
    
    print("Input File: " + args.response_file_name, file=output_file)
    if args.topk:
        evaluator.reconstrct_topk_expl(args.label_file_name)
    elif args.lime or args.occ:
        evaluator.reconstrct_topk_expl()
    else:
        evaluator.process_model_input()
        evaluator.print_fail_rate()
        evaluator.reset_fails()

    print(evaluator.explanations)
    # print(evaluator.model_labels)
    # with open('llama3_ep_expl.pickle', "wb") as handle:
    #     pickle.dump(evaluator.explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('llama3_ep_pred.pickle', "wb") as handle:
    #     pickle.dump(evaluator.model_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # accuracy = evaluator.calculate_accuracy()
    # print("Accuracy: ", str(accuracy))
    # print("Accuracy: ", str(accuracy), file=output_file)
    
    mistral_comprehensiveness = evaluator.calculate_comprehensiveness()
    print("Mistral Comprehensiveness: ", str(
        mistral_comprehensiveness), file=output_file)
    print("Mistral Comprehensiveness: ", str(
        mistral_comprehensiveness))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    mistral_sufficiency = evaluator.calculate_sufficiency()
    print("Mistral Sufficiency: ", str(mistral_sufficiency), file=output_file)
    print("Mistral Sufficiency: ", str(mistral_sufficiency))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    mistral_df_mit = evaluator.calculate_DF_MIT()
    print("Mistral DF_MIT: ", str(mistral_df_mit), file=output_file)
    print("Mistral DF_MIT: ", str(mistral_df_mit))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    mistral_df_frac = evaluator.calculate_DF_Frac()
    print("Mistral DF_Frac: ", str(mistral_df_frac), file=output_file)
    print("Mistral DF_Frac: ", str(mistral_df_frac))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    mistral_del_rank_correlation = evaluator.calculate_del_rank_correlation()
    print("Mistral Deletion Rank Correlation: ", str(
        mistral_del_rank_correlation), file=output_file)
    print("Mistral Deletion Rank Correlation: ", str(
        mistral_del_rank_correlation))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    metric_values = [mistral_comprehensiveness, mistral_sufficiency,
                     mistral_df_mit, mistral_df_frac, mistral_del_rank_correlation]
    metric_names = ["Comprehensivness", "Sufficiency",
                    "DF_MIT", "DF_Frac", "Deletion Rank Correlation"]
    metric_df = pd.DataFrame(metric_values, metric_names)
    print(metric_df, file=output_file)
    print("\nComplete!", file=output_file)
    output_file.close()