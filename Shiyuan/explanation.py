from prediction import *
from messages import *
from datasets import load_dataset
import pickle
import itertools
import ast

E_P_PROMPT = e_p_msg(few_shot=False)
P_E_PROMPT = p_e_msg(few_shot=False)

ratio = []

def reconstruct_explaination(response: list, sentence: list):
    global ratio
    word_list = []
    expl_list = []
    attribution_value ={}
    for i in sentence:
        if len(response) == 0:
            break
        elif i == response[0][0]:
            word_list.append(i)
            expl_list.append(response[0][1])
            response.pop(0)
        else:
            while i in list(dict(response).keys()):
                response.pop(0)
                if i == response[0][0]:
                    word_list.append(i)
                    expl_list.append(response[0][1])
                    response.pop(0)
                    break
    diff = [x for x in sentence if x not in word_list]
    ratio.append(len(diff)/len(sentence))
    # if len(diff) > 0:
        # print(f"The missing word / punctuation: {diff}. The ratio is {len(diff)} out of {len(sentence)} = {len(diff)/len(sentence)}.")
    for j in range(len(sentence)):
        if len(word_list) != 0 and sentence[j] == word_list[0]:
            attribution_value[j] = expl_list[0]
            word_list.pop(0)
            expl_list.pop(0)
        else:
            attribution_value[j] = 0.0
    return attribution_value

def analyze_ep_result(sentence, response):
    r = response.split("\n")
    r = [i for i in r if i.strip()]
    prediction_pair = eval(r[1])
    attribution_value = eval(r[0])
    # print("prediction pair:", prediction_pair)
    # print("attribution_value:", attribution_value)
    # if len(attribution_value) != len(sentence):
        # print(f"There are {len(attribution_value)} out of {len(sentence)} explanations being generated.")
    final_attribution_value = reconstruct_explaination(attribution_value, sentence)
    # print(final_attribution_value)
    return prediction_pair, final_attribution_value

def analyze_pe_result_P_only(sentence, response):
    r = response.split("\n")
    r = [i for i in r if i.strip()]
    prediction_pair = eval(r[0])
    # attribution_value = eval(r[1])
    # print("prediction pair:", prediction_pair)
    # print("attribution_value:", attribution_value)
    # if len(attribution_value) != len(sentence):
        # print(f"There are {len(attribution_value)} out of {len(sentence)} explanations being generated.")
    # final_attribution_value = reconstruct_explaination(attribution_value, sentence)
    # print(final_attribution_value)
    return prediction_pair

def analyze_pe_result(sentence, response):
    r = response.split("\n")
    r = [i for i in r if i.strip()]
    prediction_pair = eval(r[0])
    attribution_value = eval(r[1])
    # print("prediction pair:", prediction_pair)
    # print("attribution_value:", attribution_value)
    # if len(attribution_value) != len(sentence):
        # print(f"There are {len(attribution_value)} out of {len(sentence)} explanations being generated.")
    final_attribution_value = reconstruct_explaination(attribution_value, sentence)
    # print(final_attribution_value)
    return prediction_pair, final_attribution_value

def get_E_P_result_P_Only(sentence, prompt=E_P_PROMPT):
    prompt.append({"role": "user", "content": f"<review> {sentence} <review>"})
    # print(E_P_PROMPT)
    response = generate_response(prompt=prompt)["choices"][0]["message"]["content"]
    # print(response)
    sentence = sentence.split()
    prompt.pop()
    return analyze_pe_result_P_only(sentence, response)

def get_E_P_result(sentence, prompt=E_P_PROMPT):
    prompt.append({"role": "user", "content": f"<review> {sentence} <review>"})
    # print(E_P_PROMPT)
    response = generate_response(prompt=prompt)["choices"][0]["message"]["content"]
    # print(response)
    sentence = sentence.split()
    prompt.pop()
    return analyze_ep_result(sentence, response)
    

def get_P_E_result(sentence, prompt=P_E_PROMPT):
    prompt.append({"role": "user", "content": f"<review>{sentence}<review>"})
    response = generate_response(prompt=prompt)["choices"][0]["message"]["content"]
    # print(response)
    sentence = sentence.split()
    prompt.pop()
    return analyze_pe_result(sentence, response)

def explain_then_predict(dataset):
    sentence = []
    prediction = []
    attribute_value = []
    for i in dataset:
        print(f"** {dataset.index(i)} ********************************************")
        print(i)
        sentence.append(i)
        ep_prediction, ep_attribute_value = get_E_P_result(i)
        prediction.append(ep_prediction)
        attribute_value.append(ep_attribute_value)
    with open("EP_Result", "wb") as dbfile:
        pickle.dump({"sentence": sentence, "prediction": prediction, "saliency list": attribute_value}, dbfile)

def predict_and_explain(dataset):
    sentence = []
    prediction = []
    attribute_value = []
    for i in dataset:
        sentence.append(i)
        print(f"** {dataset.index(i)} ********************************************")
        print(i)
        pe_prediction, pe_attribute_value = get_P_E_result(i)
        prediction.append(pe_prediction)
        attribute_value.append(pe_attribute_value)
    with open("PE_Result", "wb") as dbfile:
        pickle.dump({"sentence": sentence, "prediction": prediction, "saliency list": attribute_value}, dbfile)

def main():
    dataset = load_dataset('sst', split='test')
    size = 100
    dataset = dataset.shuffle(seed=8)['sentence']

    # explain_then_predict(dataset[:size:])
    predict_and_explain(dataset[:size:])
    # print(f"The mean ratio is {sum(ratio)/len(ratio)}")

if __name__ == "__main__":
    main()