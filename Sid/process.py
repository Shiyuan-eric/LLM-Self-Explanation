import ast
from tqdm import tqdm
import openai
from retry import retry
import pickle
import random
import time
import pandas as pd
from scipy import stats

openai.api_key = "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"



# Model parameters for repeatability
# Testing only PE for now
# messages = [
#     {
#       "role": "system",
#       "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
#     }
#   ]
TEMPERATURE = 0
MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1024
TOP_P = 1
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
PRE_PHRASE = "<review> "
POST_PHRASE = " <review>"



# For random number generation
random_range = 10e-4
random.seed(0)


class GPT_Evaluator():
    


    def __init__(self, response_filename, PE, messages, label_filename):
        self.messages = messages
        self.fails = 0
        self.total = 0
        self.ovr_fails = 0
        self.ovr_total = 0
        self.response_filename = response_filename
        self.PE = PE
        self.label_filename = label_filename

    def convert_from_Shiyuan_format(self, inp, sentence):
        explanations = []
        tokens = sentence.split(' ')
        for key, value in inp.items():
            index = key
            saliency = value
            word = tokens[index]
            new_exp = ((word, (saliency, index)), tokens)
            explanations.append(new_exp)
        return explanations

    def process_GPT_input_Shiyuan(self):
        with open(self.response_filename, 'rb') as handle:
            responses = pickle.load(handle)
        
    
    def process_GPT_input(self):
        with open(self.response_filename, 'rb') as handle:
            responses = pickle.load(handle)
        self.explanations = []
        self.gpt_labels = []
        self.sentences = []
        for response in responses.items():
            (prediction, confidence, exp) = self.parse_completion(response[1])
            self.sentences.append(response[0])
            self.gpt_labels.append((prediction, confidence))
            new_exp = []
            orig_tokens = response[0].split(' ')
            gpt_tokens = []
            for i in range(len(exp)):
                gpt_tokens.append(exp[i][0])
                
            for i in range(len(orig_tokens)):
                try:
                    idx = gpt_tokens.index(orig_tokens[i])
                except:
                    idx = -1
                if idx != -1:
                    new_exp.append((gpt_tokens[idx], (exp[idx][1] + random.uniform(-1 * random_range, random_range), i)))
                    gpt_tokens[idx] = ''
                else:
                    new_exp.append((orig_tokens[i], (random.uniform(-1 * random_range, random_range), i)))
            
            new_exp = sorted(new_exp, key=lambda x: x[1][0], reverse=True)
            self.explanations.append((new_exp, orig_tokens))
        
    def generate_response(self):
        while True:
            try:
                response = openai.ChatCompletion.create(
                model=MODEL,
                messages=self.messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY
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
    
    def reset_fails(self):
        self.ovr_fails += self.fails
        self.ovr_total += self.total
        self.fails = 0
        self.total = 0

    def parse_completion(self, response):
        lines = response.splitlines()
        self.total += 1
        try:
            if self.PE:
                exp = ast.literal_eval(lines[1])
                (prediction, confidence) = ast.literal_eval(lines[0])
            else:
                exp = ast.literal_eval(lines[0])
                (prediction, confidence) = ast.literal_eval(lines[1])
        except:
            if not self.PE:
                try:
                   # Trying to see if the potential error was that there was a newline(something I saw a few times)
                   exp = ast.literal_eval(lines[0])
                   prediction, confidence = ast.literal_eval(lines[2])
                   return (prediction, confidence, exp)
                except:
                    pass
            # GPT didn't give an answer in the required format (more likely an invalid response)
            # So, make everything 0
            exp = []
            for token in response.split(' '):
                exp.append((token, 0.0))
            (prediction, confidence) = (0, 0.5)
            self.fails += 1
        return (prediction, confidence, exp)
    
    def calculate_accuracy(self):
        print("Calculating Accuracy...")
        with open(self.label_filename, 'rb') as handle:
            labels = pickle.load(handle)
        correct = 0.0
        total = 0.0
        for i in tqdm(range(len(self.gpt_labels))):
            lb = (labels[i] - 0.5 > 0)
            if(self.gpt_labels[i][0] == lb):
                correct += 1
            total += 1
            
        return correct / total

    def get_completion(self):
        gpt_response = self.generate_response().choices[0].message.content
        return self.parse_completion(gpt_response)

    def calculate_comprehensiveness(self):
        print("Calculating Comprehensivness...")
        compr_list = []
        for i in tqdm(range(len(self.explanations))):
            ret = 0
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = tkns[:]
            phrase = ' '.join(tkns)
            self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            self.messages.pop()
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                word = tkns[index]
                masked_tkns[index] = ''
                phrase = ' '.join([t for t in masked_tkns if t != ''])
                if(phrase == ""):
                    ret += pre_y_prob - 0.5
                    continue
                self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
                (y_label, y_prob, __) = self.get_completion()
                self.messages.pop()
                pre_y_prob = pre_y_prob if pre_y_label == 1 else (1 - pre_y_prob)
                y_prob = y_prob if y_label == 1 else (1 - y_prob)
                ret += pre_y_prob - y_prob
            # Add empty phrase as well:
            
            ret = ret / (len(explanation) + 1)
            compr_list.append(ret)
        return self.calculate_avg(compr_list)
    
    def calculate_sufficiency(self):
        print("Calculating Sufficiency...")
        suff_list = []
        for i in tqdm(range(len(self.explanations))):
            ret = 0
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = ['' for token in tkns]
            phrase = ' '.join(tkns)
            self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            self.messages.pop()
            pre_y_prob = pre_y_prob if pre_y_label == 1 else (1 - pre_y_prob)
            y_prob = 0.5
            ret += abs(pre_y_prob - y_prob)
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                masked_tkns[index] = tkns[index]
                phrase = ' '.join([t for t in masked_tkns if t != ''])
                self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
                (y_label, y_prob, __) = self.get_completion()
                self.messages.pop()
                y_prob = y_prob if y_label == 1 else (1 - y_prob)
                ret += abs(pre_y_prob - y_prob)

            ret = ret / (len(explanation) + 1)
            suff_list.append(ret)

        return suff_list

    def calculate_avg(self, cmp_list):
        return sum(cmp_list) / len(cmp_list)

    # DF_MIT
    def calculate_DF_MIT(self):
        print("Calculating DF_MIT...")
        flipped = 0
        total = 0
        for i in tqdm(range(len(self.explanations))):
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            phrase = ' '.join(tkns)
            self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            self.messages.pop()
            if(len(explanation) == 0):
                continue
            index = explanation[0][1][1]
            word = tkns.pop(index)
            phrase = ' '.join(tkns)
            tkns.insert(index, word)
            self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
            (y_label, y_prob, __) = self.get_completion()
            self.messages.pop()
            if y_label != pre_y_label:
                flipped += 1
            total += 1
        return flipped/total
    
    # DF_Frac
    def calculate_DF_Frac(self):
        print("Calculating DF_Frac...")
        frac_list = []

        for i in tqdm(range(len(self.explanations))):
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = [token for token in tkns]
            phrase = ' '.join(tkns)
            self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            self.messages.pop()
            num_taken = len(explanation) + 1
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                masked_tkns[index] = ''
                phrase = ' '.join([t for t in masked_tkns if t != ''])
                self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
                (y_label, y_prob, __) = self.get_completion()
                self.messages.pop()
                # Need to account for GPT misinput, make it be the same
                if y_label != pre_y_label:
                    num_taken = j + 1
                    break
            frac_list.append(num_taken/(len(explanation) + 1))

        return self.calculate_avg(frac_list)

    def calculate_del_rank_correlation(self):
        print("Calculating Deletion Rank Correlation...")
        ret_list = []
        for i in tqdm(range(len(self.explanations))):
            explanation = self.explanations[i][0][:]
            tkns = self.explanations[i][1][:]
            phrase = ' '.join(tkns)
            self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            self.messages.pop()
            delta_f = []
            e = []
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                word = tkns[index]
                tkns[index] = ""
                self.messages.append({"role": "user", "content": PRE_PHRASE + phrase + POST_PHRASE})
                (y_label, y_prob, __) = self.get_completion()
                self.messages.pop()
                delta_f.append(pre_y_prob - y_prob + random.uniform(-1 * random_range, random_range))
                e.append(explanation[j][1][0] + random.uniform(-1 * random_range, random_range))
                tkns[index] = word
            # print(delta_f)
            # print(e)
            ret_list.append((stats.spearmanr(delta_f, e)).correlation)
            # print(ret_list)
        return self.calculate_avg(ret_list)
    
    def print_fail_rate(self):
        print("Fails/Total = %d/%d = %.2f%%" % (self.fails, self.total, float(self.fails / self.total)))

# tqdm(range(len(self.explanations)))

if __name__ == "__main__":
    response_filename = "gpt_response_EP.pickle"
    label_filename = "labels_EP.pickle"
    #EP
    messages = [
        {
            "role": "system",
            "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
        }
    ]
    evaluator = GPT_Evaluator(response_filename=response_filename, PE=False, messages=messages, label_filename=label_filename)

    

    print("Input File: " + response_filename)
    evaluator.process_GPT_input()
    evaluator.print_fail_rate()
    evaluator.reset_fails()
    
    accuracy = evaluator.calculate_accuracy()
    print("Accuracy: ", str(accuracy))

    gpt_comprehensiveness = evaluator.calculate_comprehensiveness()
    print("GPT Comprehensiveness: ", str(gpt_comprehensiveness))
    evaluator.print_fail_rate()
    evaluator.reset_fails()

    gpt_sufficiency = evaluator.calculate_sufficiency()
    print("GPT Sufficiency: ", str(gpt_sufficiency))
    evaluator.print_fail_rate()
    evaluator.reset_fails()

    gpt_df_mit = evaluator.calculate_DF_MIT()
    print("GPT DF_MIT: ", str(gpt_df_mit))
    evaluator.print_fail_rate()
    evaluator.reset_fails()

    gpt_df_frac = evaluator.calculate_DF_Frac()
    print("GPT DF_Frac: ", str(gpt_df_frac))
    evaluator.print_fail_rate()
    evaluator.reset_fails()

    gpt_del_rank_correlation = evaluator.calculate_del_rank_correlation()
    print("GPT Deletion Rank Correlation: ", str(gpt_del_rank_correlation))
    evaluator.print_fail_rate()
    evaluator.reset_fails()

    metric_values = [accuracy, gpt_comprehensiveness, gpt_sufficiency,
                     gpt_df_mit, gpt_df_frac, gpt_del_rank_correlation]
    metric_names = ["Accuracy", "Comprehensivness", "Sufficiency", "DF_MIT", "DF_Frac", "Deletion Rank Correlation"]
    metric_df = pd.DataFrame(metric_values, metric_names)
    print(metric_df)
    print("\nComplete!")
