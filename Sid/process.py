import ast
from tqdm import tqdm
import openai
from retry import retry
import pickle
import random

openai.api_key = "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"


# Model parameters for repeatability    
# Testing only PE for now
MESSAGES = [
    {
      "role": "system",
      "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. \n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    }
  ]
TEMPERATURE = 0
MODEL = "gpt-3.5-turbo"
PRE_PHASE = "This is the sentence: "

# For random number generation
random_range = 10e-5
random.seed(0)


class GPT_Evaluator():
    def __init__(self, filename, PE):
        self.response_file = open(filename, "r")
        self.PE = PE
        
    def skip_testing(self):
        with open('sentences.pickle', 'rb') as handle:
            sentences = pickle.load(handle)
        for i in range(len(self.explanations)):
            e = self.explanations[i][0]
            print("Explanation words: %d, Actual words: %d, ratio: %f" %(len(e), len(sentences[i].split(), len(e)/len(sentences[i].split()))))

    # This function doesn't work for the current format
    def test_refuse_to_answer(self):
        ret_list = []
        # t = []
        for i in tqdm(range(len(self.explanations))):
            ret = 0
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = tkns[:]
            phrase = ' '.join(tkns)
            messages = MESSAGES[:]
            messages.append({"role": "user", "content": PRE_PHASE + phrase})
            completion = self.get_completion(messages)
            messages.pop()
            num_success = len(explanation)
            print("Original sentence: ", phrase)
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                masked_tkns[index] = ''
                phrase = ' '.join([t for t in masked_tkns if t != ''])
                messages.append({"role": "user", "content": PRE_PHASE + phrase})
                completion = self.get_completion(messages)
                messages.pop()
                (y_label, y_prob, __) = self.parse_completion(completion)
            # print("Worked for %d out of %d cases. Ratio is %.4f" % (num_success, len(explanation), float(num_success/len(explanation))))
            ret_list.append(ret)
        print(ret_list)
        return self.calculate_avg(ret_list)
    
    def process_GPT_input(self):
        with open('gpt_response.pickle', 'rb') as handle:
            responses = pickle.load(handle)
        self.explanations = []
        self.gpt_labels = []
        for response in responses:
            (prediction, confidence, exp) = self.parse_completion(response)

            self.gpt_labels.append((prediction, confidence))
            tkns = []
            new_exp = []
            for i in range(len(exp)):
                tkns.append(exp[i][0])
                new_exp.append((exp[i][0], (exp[i][1] + random.uniform(-1 * random_range, random_range), i)))
            
            new_exp = sorted(new_exp, key=lambda x: x[1][0], reverse=True)
            
            self.explanations.append((new_exp, tkns))
        
        # for expl in self.explanations:
            # print(expl)
        
    @retry(tries=2, delay=1)  
    def get_completion(self, messages):
        completion = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=messages,
                    temperature=TEMPERATURE
                )
        return completion

    def parse_completion(self, response):
        # print(response)
        lines = response.splitlines()
        try:
            if self.PE:
                exp = ast.literal_eval(lines[1])
                (prediction, confidence) = ast.literal_eval(lines[0])
            else:
                exp = ast.literal_eval(lines[0])
                (prediction, confidence) = ast.literal_eval(lines[1])
        except:
            # GPT didn't understand/format strayed. Go in the middle.
            exp = []
            for token in response.split(' '):
                exp.append((token, 0.0))
            (prediction, confidence) = (0, 0.5)
            print("Failed for " + response)
        return (prediction, confidence, exp)

    def calculate_comprehensiveness(self):
        compr_list = []
        for i in tqdm(range(len(self.explanations))):
            ret = 0
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = tkns[:]
            phrase = ' '.join(tkns)
            messages = MESSAGES[:]
            messages.append({"role": "user", "content": PRE_PHASE + phrase})
            completion = self.get_completion(messages)
            messages.pop()
            (pre_y_label, pre_y_prob, __) = self.parse_completion(completion.choices[0].message.content)
            # pre_y = completion()
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                word = tkns[index]
                masked_tkns[index] = ''
                phrase = ' '.join([t for t in masked_tkns if t != ''])
                if(phrase == ""):
                    ret += pre_y_prob - 0.5
                    continue
                messages.append({"role": "user", "content": PRE_PHASE + phrase})
                completion = self.get_completion(messages)
                messages.pop()
                (y_label, y_prob, __) = self.parse_completion(completion.choices[0].message.content)
                pre_y_prob = pre_y_prob if pre_y_label == 1 else (1 - pre_y_prob)
                y_prob = y_prob if y_label == 1 else (1 - y_prob)
                ret += pre_y_prob - y_prob
            # Add empty phrase as well:
            
            ret = ret / (len(explanation) + 1)
            compr_list.append(ret)
        return self.calculate_avg(compr_list)
    
    def calculate_sufficiency(self):
        suff_list = []
        for i in tqdm(range(len(self.explanations))):
            ret = 0
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = ['' for token in tkns]
            phrase = ' '.join(tkns)
            messages = MESSAGES[:]
            messages.append({"role": "user", "content": PRE_PHASE + phrase})
            completion = self.get_completion(messages)
            messages.pop()
            (pre_y_label, pre_y_prob, __) = self.parse_completion(completion.choices[0].message.content)
            pre_y_prob = pre_y_prob if pre_y_label == 1 else (1 - pre_y_prob)
            y_prob = 0.5
            ret += abs(pre_y_prob - y_prob)
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                masked_tkns[index] = tkns[index]
                phrase = ' '.join([t for t in masked_tkns if t != ''])
                messages.append({"role": "user", "content": "Now the next sentence: " + phrase})
                completion = self.get_completion(messages)
                messages.pop()
                (y_label, y_prob, __) = self.parse_completion(completion.choices[0].message.content)
                y_prob = y_prob if y_label == 1 else (1 - y_prob)
                ret += abs(pre_y_prob - y_prob)

            ret = ret / (len(explanation) + 1)
            suff_list.append(ret)

        return suff_list

    def calculate_avg(self, cmp_list):
        return sum(cmp_list) / len(cmp_list)

    # DF_MIT
    def calculate_DF_MIT(self):
        flipped = 0
        total = 0
        for i in tqdm(range(len(self.explanations))):
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            phrase = ' '.join(tkns)
            messages = MESSAGES[:]
            messages.append({"role": "user", "content": PRE_PHASE + phrase})
            completion = self.get_completion(messages)
            messages.pop()
            (pre_y_label, pre_y_prob, __) = self.parse_completion(completion.choices[0].message.content)
            if(len(explanation) == 0):
                continue
            index = explanation[0][1][1]
            word = tkns.pop(index)
            phrase = ' '.join(tkns)
            tkns.insert(index, word)
            messages.append({"role": "user", "content": PRE_PHASE + phrase})
            completion = self.get_completion(messages)
            messages.pop()
            (y_label, y_prob, __) = self.parse_completion(completion.choices[0].message.content)
            if y_label != pre_y_label:
                flipped += 1
            total += 1
        return flipped/total
    
    # DF_Frac
    def calculate_DF_Frac(self):
        frac_list = []

        for i in tqdm(range(len(self.explanations))):
            explanation = self.explanations[i][0]
            tkns = self.explanations[i][1][:]
            masked_tkns = [token for token in tkns]
            phrase = ' '.join(tkns)
            messages = MESSAGES[:]
            messages.append({"role": "user", "content": PRE_PHASE + phrase})
            completion = self.get_completion(messages)
            messages.pop()
            (pre_y_label, pre_y_prob, __) = self.parse_completion(completion.choices[0].message.content)
            num_taken = len(explanation) + 1
            for j in range(len(explanation)):
                index = explanation[j][1][1]
                masked_tkns[index] = ''
                phrase = ' '.join([t for t in masked_tkns if t != ''])
                messages.append({"role": "user", "content": PRE_PHASE + phrase})
                completion = self.get_completion(messages)
                messages.pop()
                # Need to account for GPT misinput, make it be the same
                (y_label, y_prob, __) = self.parse_completion(completion.choices[0].message.content)
                if y_label != pre_y_label:
                    num_taken = j + 1
                    break
            frac_list.append(num_taken/(len(explanation) + 1))

        return self.calculate_avg(frac_list)

    def calculate_del_rank_correlation(self):
        # To be implemented
        pass

if __name__ == "__main__":
    evaluator = GPT_Evaluator("gpt_response2.txt", PE=True)
    evaluator.process_GPT_input()
    # gpt_comprehensiveness = evaluator.calculate_comprehensiveness()
    # print("GPT Comprehensiveness: ", str(gpt_comprehensiveness))
    # gpt_sufficiency = evaluator.calculate_sufficiency()
    # print("GPT Sufficiency: ", str(gpt_sufficiency))
    gpt_df_mit = evaluator.calculate_DF_MIT()
    print("GPT DF_MIT: ", str(gpt_df_mit))
    gpt_df_frac = evaluator.calculate_DF_Frac()
    print("GPT DF_Frac: ", str(gpt_df_frac))
    # evaluator.skip_testing()
