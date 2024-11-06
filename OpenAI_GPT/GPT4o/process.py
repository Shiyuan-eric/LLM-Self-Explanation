import time
import os
import sys
import getopt
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import pandas as pd

from model_evaluator import Model_Evaluator

def loadData(filename):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
def storeData(filename, data):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-r', '--response_filename', type=str, help='', required=True)
    parser.add_argument('-o', '--output_file', type=str, help='')
    parser.add_argument('-occ', action='store_true', help='')
    parser.add_argument('-lime', action='store_true', help='')
    parser.add_argument('-pe', action='store_true', help='')
    parser.add_argument('-p_only', action='store_true')
    parser.add_argument('-topk', action='store_true')
    args = parser.parse_args()

    label_filename = "true_labels.pickle"
    if args.output_file:
        output_file = open(args.output_file, 'w')
    else:
        output_file = sys.stdout
    
    evaluator = Model_Evaluator(response_filename=args.response_filename, PE=args.pe, label_filename=label_filename, traditionalExpl=(args.occ or args.lime or args.topk), p_only=args.p_only)
    evaluator.reconstruct_expl()
    if not args.occ and not args.lime and not args.topk:
        if args.pe:
            storeData('gpt4o_pe_expl.pickle', evaluator.explanations)
            storeData('gpt4o_pe_pred.pickle', evaluator.model_labels)
            evaluator.cache_results("gpt4o_pe_cache.pickle")
        else:
            storeData('gpt4o_ep_expl.pickle', evaluator.explanations)
            storeData('gpt4o_ep_pred.pickle', evaluator.model_labels)
            evaluator.cache_results("gpt4o_ep_cache.pickle")
            evaluator.print_fail_rate()
            evaluator.reset_fails()
    
        accuracy = evaluator.calculate_accuracy()
        print("Accuracy: ", str(accuracy))
        print("Accuracy: ", str(accuracy), file=output_file)
    elif args.topk:
        print("TOP-K")
        if args.pe:
            evaluator.model_labels = loadData("gpt4o_topk_labels_PE.pickle")
        else:
            evaluator.model_labels = loadData("gpt4o_topk_labels_EP.pickle")
        accuracy = evaluator.calculate_accuracy()
        print("Accuracy: ", str(accuracy))
        print("Accuracy: ", str(accuracy), file=output_file)

    
    comprehensiveness = evaluator.calculate_comprehensiveness()
    print("Comprehensiveness: ", str(
        comprehensiveness), file=output_file)
    print("Comprehensiveness: ", str(
        comprehensiveness))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    sufficiency = evaluator.calculate_sufficiency()
    print("Sufficiency: ", str(sufficiency), file=output_file)
    print("Sufficiency: ", str(sufficiency))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    df_mit = evaluator.calculate_DF_MIT()
    print("DF_MIT: ", str(df_mit), file=output_file)
    print("DF_MIT: ", str(df_mit))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    df_frac = evaluator.calculate_DF_Frac()
    print("DF_Frac: ", str(df_frac), file=output_file)
    print("DF_Frac: ", str(df_frac))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    del_rank_correlation = evaluator.calculate_del_rank_correlation()
    print("Deletion Rank Correlation: ", str(
        del_rank_correlation), file=output_file)
    print("Deletion Rank Correlation: ", str(
        del_rank_correlation))
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    metric_values = [comprehensiveness, sufficiency,
                     df_mit, df_frac, del_rank_correlation]
    metric_names = ["Comprehensivness", "Sufficiency",
                    "DF_MIT", "DF_Frac", "Deletion Rank Correlation"]
    metric_df = pd.DataFrame(metric_values, metric_names)
    print(metric_df, file=output_file)
    print("\nComplete!", file=output_file)
    output_file.close()