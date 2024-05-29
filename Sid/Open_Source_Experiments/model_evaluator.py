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
import re

llama_P_MSG = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)"
    }
    ]

llama_P_E_MSG = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    }
    ]

llama_E_P_MSG = [
    {
        "role": "user",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
    }
    ]

mistral_P_MSG = [
    {
        "role": "user",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
    ]

mistral_P_E_MSG = [
    {
        "role": "user",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. Each word or punctuation is separated by a space.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
    ]

mistral_E_P_MSG = [
    {
        "role": "user",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nThe movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of the movie review.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
    },
    {
        "role": "assistant", "content": "I understand. Please send a review and I will do my best to respond in the desired format."
    }
    ]


class Model_Evaluator():

    def __init__(self, model, tokenizer, response_filename, PE, label_filename, max_tokens, p_only, mistral, llama):

        # self.messages = messages
        self.tokenizer = tokenizer
        self.model = model
        self.max_tokens = max_tokens
        self.fails = 0
        self.total = 1
        self.ovr_fails = 0
        self.ovr_total = 1
        self.response_filename = response_filename
        self.PE = PE
        self.p_only = p_only
        self.mistral = mistral
        self.llama = llama
        if self.p_only:
            print("p_only mode.")
            if self.mistral:
                self.messages = mistral_P_MSG.copy()
            elif self.llama:
                self.messages = llama_P_MSG.copy()
        elif self.PE:
            if self.mistral:
                self.messages = mistral_P_E_MSG.copy()
            elif self.llama:
                self.messages = llama_P_E_MSG.copy()
        else:
            if self.mistral:
                self.messages = mistral_E_P_MSG.copy()
            elif self.llama:
                self.messages = llama_E_P_MSG.copy()
        self.label_filename = label_filename

        self.pre_phrase = "<review> "
        self.post_phrase = " <review>"
        
        self.random_range = 10e-4
        random.seed(0)

    # Opens LIME responses from pickle file
    def process_LIME_input(self):
        with open(self.response_filename, 'rb') as handle:
            explanations = pickle.load(handle)
        self.explanations = explanations
    
    # Parses model response into tuple containing model's prediction, confidence, and explanation
    def parse_completion(self, response, sentence):
        lines = response.splitlines()
        lines = [string for string in lines if string]
        lines = [string for string in lines if re.search(r'\d', string)]
        self.total += 1
        try:
            if self.p_only:
                (prediction, confidence) = ast.literal_eval(lines[0])
                exp = None
            elif self.PE:
                exp = ast.literal_eval(lines[1])
                cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[0])
                (prediction, confidence) = ast.literal_eval(cleaned_string)
            else:
                exp = ast.literal_eval(lines[0])
                cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[1])
                (prediction, confidence) = ast.literal_eval(cleaned_string)
        except:
            if not self.PE:
                try:
                    # Trying to see if the potential error was that there was a newline(something I saw a few times)
                    exp = ast.literal_eval(lines[0])
                    cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[2])
                    prediction, confidence = ast.literal_eval(cleaned_string)
                    return (prediction, confidence, exp)
                except:
                    pass
            # GPT didn't give an answer in the required format (more likely an invalid response)
            # So, make everything 0
            exp = []
            for token in sentence.split(' '):
                exp.append((token, 0.0))
            (prediction, confidence) = (0, 0.5)
            self.fails += 1
        return (prediction, confidence, exp)

    def parse_completion_no_exp(self, response):
        lines = response.splitlines()
        lines = [string for string in lines if string]
        lines = [string for string in lines if re.search(r'\d', string)]
        self.total += 1
        try:
            if self.p_only:
                cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[0])
                (prediction, confidence) = ast.literal_eval(cleaned_string)
            elif self.PE:
                cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[0])
                (prediction, confidence) = ast.literal_eval(cleaned_string)
            else:
                cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[1])
                (prediction, confidence) = ast.literal_eval(cleaned_string)
        except:
            if not self.PE:
                try:
                    # Trying to see if the potential error was that there was a newline(something I saw a few times)                    
                    cleaned_string = re.sub(r'[^0-9,.()]+', '', lines[2])
                    prediction, confidence = ast.literal_eval(cleaned_string)
                    return (prediction, confidence, None)
                except:
                    pass
            # GPT didn't give an answer in the required format (more likely an invalid response)
            # So, make everything 0
            (prediction, confidence) = (0, 0.5)
            self.fails += 1
        return (prediction, confidence, None)

    # Processes the raw mixtral responses into the explanation format
    def process_model_input(self):
        with open(self.response_filename, 'rb') as handle:
            responses = pickle.load(handle)
        # print(responses)
        # iter = 3
        # for i, value in enumerate(responses.values()):
        #     if i < 3:
        #         print(value)
        #     else:
        #         break
        self.explanations = []
        self.model_labels = []
        self.sentences = []
        for response in responses.items():
            (prediction, confidence, exp) = self.parse_completion(response[1], response[0])
            self.sentences.append(response[0])
            self.model_labels.append((prediction, confidence))
            new_exp = []
            orig_tokens = response[0].split(' ')
            model_tokens = []
            try:
                for i in range(len(exp)):
                    model_tokens.append(exp[i][0])
            except:
                print("Something is wrong!")
                print(response[0], exp)
                model_tokens = []
                sentence_list = response[0].split()
                for i in range(len(sentence_list)):
                    model_tokens.append((sentence_list[i], 0.0))
                print(model_tokens)
                print("THE END")


            # match gpt token saliency values to original tokens (small epsilon if no match)
            for i in range(len(orig_tokens)):
                try:
                    idx = model_tokens.index(orig_tokens[i])
                except:
                    idx = -1
                if idx != -1:
                    new_exp.append(
                        (model_tokens[idx], (exp[idx][1] + random.uniform(0, self.random_range), i)))
                    model_tokens[idx] = ''
                else:
                    new_exp.append(
                        (orig_tokens[i], (random.uniform(0, self.random_range), i)))

            new_exp = sorted(new_exp, key=lambda x: x[1][0], reverse=True)
            self.explanations.append((new_exp, orig_tokens))
    def reconstrct_topk_expl(self, label_file_name=None):
        with open(self.response_filename, 'rb') as f:
            self.explanations = pickle.load(f)
        if label_file_name:
            with open(label_file_name, 'rb') as f:
                self.model_labels = pickle.load(f)

    def reconstruct_expl(self):
        with open(self.response_filename, 'rb') as f:
            occlusion_pickle = pickle.load(f)
        expl_list = []
        for sentence, occlusion_dict in occlusion_pickle.items():
            sentence_list = sentence.split()
            expl = []
            for index, attribute_value in occlusion_dict.items():
                expl.append((sentence_list[index], (attribute_value, index)))
            expl_list.append((expl, sentence_list))
        self.explanations = expl_list.copy()

    # Function to query the openai API and generate a gpt response given a prompt
    def generate_mistral_response(self):
        inputs = self.tokenizer.apply_chat_template(
            self.messages, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.max_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        last_instruction_idx = response.rindex("[/INST]") + 7
        return response[last_instruction_idx:]

    def generate_llama_response(self):
        input_ids = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            pad_token_id = self.tokenizer.eos_token_id,
            eos_token_id=terminators,
            # do_sample=True,
            # temperature=0.6,
            # top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return(self.tokenizer.decode(response, skip_special_tokens=True))

    def generate_response(self):
        if self.mistral:
            return self.generate_mistral_response()
        elif self.llama:
            return self.generate_llama_response()

    # Reset
    def reset_fails(self):
        self.ovr_fails += self.fails
        self.ovr_total += self.total
        self.fails = 0
        self.total = 1

    # Calculates accuracy
    def calculate_accuracy(self):
        print("Calculating Accuracy...")
        with open(self.label_filename, 'rb') as handle:
            labels = pickle.load(handle)
        correct = 0.0
        total = 0.0
        for i in tqdm(range(len(self.model_labels))):
            lb = (labels[i] - 0.5 > 0)
            if (self.model_labels[i][0] == lb):
                correct += 1
            total += 1

        return correct / total

    # Generates and parses model response
    def get_completion(self):
        model_response = self.generate_response()
        return self.parse_completion_no_exp(model_response)

    # Implements comprehensiveness metric (DeYoung et al., 2020)
    def calculate_comprehensiveness(self):
        print("Calculating Comprehensivness...")
        compr_list = []
        for i in tqdm(range(len(self.explanations))):
            ret = 0
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = tkns[:]
            phrase = ' '.join(tkns)
            self.messages.append(
                {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            self.messages.pop()
            for j in range(len(explanation)):
                # remove the tokens one by one
                index = explanation[j][1][1]
                word = tkns[index]
                masked_tkns[index] = ''
                phrase = ' '.join([t for t in masked_tkns if t != ''])

                # All tokens removed
                if (phrase == ""):
                    ret += pre_y_prob - 0.5
                    continue
                self.messages.append(
                    {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
                (y_label, y_prob, __) = self.get_completion()
                self.messages.pop()
                pre_y_prob = pre_y_prob if pre_y_label == 1 else (
                    1 - pre_y_prob)
                y_prob = y_prob if y_label == 1 else (1 - y_prob)
                ret += pre_y_prob - y_prob
            # Add empty phrase as well so +1
            ret = ret / (len(explanation) + 1)
            compr_list.append(ret)
        return self.calculate_avg(compr_list)

    # Implements sufficiency metric (DeYoung et al., 2020)
    def calculate_sufficiency(self):
        print("Calculating Sufficiency...")
        suff_list = []
        for i in tqdm(range(len(self.explanations))):
            ret = 0
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = ['' for token in tkns]
            phrase = ' '.join(tkns)
            self.messages.append(
                {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            self.messages.pop()
            pre_y_prob = pre_y_prob if pre_y_label == 1 else (1 - pre_y_prob)
            y_prob = 0.5
            ret += abs(pre_y_prob - y_prob)
            for j in range(len(explanation)):
                # add tokens back one by one
                index = explanation[j][1][1]
                masked_tkns[index] = tkns[index]
                phrase = ' '.join([t for t in masked_tkns if t != ''])

                self.messages.append(
                    {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
                (y_label, y_prob, __) = self.get_completion()
                self.messages.pop()
                y_prob = y_prob if y_label == 1 else (1 - y_prob)
                ret += abs(pre_y_prob - y_prob)

            ret = ret / (len(explanation) + 1)
            suff_list.append(ret)

        return self.calculate_avg(suff_list)

    # Returns average of a list
    def calculate_avg(self, cmp_list):
        return sum(cmp_list) / len(cmp_list)

    # Implements DF_MIT metric (Chrysostomou and Ale- tras, 2021)
    def calculate_DF_MIT(self):
        print("Calculating DF_MIT...")
        flipped = 0
        total = 0
        for i in tqdm(range(len(self.explanations))):
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            phrase = ' '.join(tkns)
            og_phrase = phrase
            self.messages.append(
                {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            npre_y_prob = pre_y_prob if pre_y_label == 1 else (1 - pre_y_prob)
            self.messages.pop()
            if (len(explanation) == 0):
                continue
            index = explanation[0][1][1]
            word = tkns.pop(index)

            phrase = ' '.join(tkns)  # phrase w/o most important token
            tkns.insert(index, word)  # restore tokens
            self.messages.append(
                {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
            (y_label, y_prob, __) = self.get_completion()
            ny_prob = y_prob if y_label == 1 else (1 - y_prob)
            self.messages.pop()

            if y_label != pre_y_label:
                # print("(%d, %f) --> (%d, %f)" %
                #   (pre_y_label, pre_y_prob, y_label, y_prob))
                if ((ny_prob > 0.5 and npre_y_prob <= 0.5) or (ny_prob <= 0.5 and npre_y_prob > 0.5)):
                    flipped += 1
            elif ((ny_prob > 0.5 and npre_y_prob <= 0.5) or (ny_prob <= 0.5 and npre_y_prob > 0.5)):
                print("(%d, %f) --> (%d, %f)" %
                      (pre_y_label, pre_y_prob, y_label, y_prob))
                flipped += 1
            total += 1
        return flipped/total

    # Implements DF_Frac metric (Serrano and Smith, 2019)
    def calculate_DF_Frac(self):
        print("Calculating DF_Frac...")
        frac_list = []

        for i in tqdm(range(len(self.explanations))):
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = [token for token in tkns]
            phrase = ' '.join(tkns)
            self.messages.append(
                {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            self.messages.pop()
            num_taken = len(explanation) + 1
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                masked_tkns[index] = ''
                phrase = ' '.join([t for t in masked_tkns if t != ''])
                self.messages.append(
                    {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
                (y_label, y_prob, __) = self.get_completion()
                self.messages.pop()
                if y_label != pre_y_label:
                    num_taken = j + 1  # took j+1 tokens to flip decision
                    break
            frac_list.append(num_taken/(len(explanation) + 1))

        return self.calculate_avg(frac_list)

    # Implements deletion rank correlation metric (Alvarez-Melis and Jaakkola, 2018b)
    def calculate_del_rank_correlation(self):
        print("Calculating Deletion Rank Correlation...")
        ret_list = []
        for i in tqdm(range(len(self.explanations))):
            explanation = self.explanations[i][0][:]
            tkns = self.explanations[i][1][:]
            phrase = ' '.join(tkns)
            self.messages.append(
                {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
            (pre_y_label, pre_y_prob, __) = self.get_completion()
            self.messages.pop()
            delta_f = []
            e = []
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                word = tkns[index]
                tkns[index] = ""
                self.messages.append(
                    {"role": "user", "content": self.pre_phrase + phrase + self.post_phrase})
                (y_label, y_prob, __) = self.get_completion()
                self.messages.pop()
                delta_f.append(pre_y_prob - y_prob +
                               random.uniform(-1 * self.random_range, self.random_range))
                e.append(
                    explanation[j][1][0] + random.uniform(-1 * self.random_range, self.random_range))
                tkns[index] = word

            # spearman coeff between delta_f and e
            ret_list.append((stats.spearmanr(delta_f, e)).correlation)
        return self.calculate_avg(ret_list)

    def print_fail_rate(self):
        print("Fails/Total = %d/%d = %.2f" %
              (self.fails, self.total, float(self.fails / self.total)))
