import random
from datasets import load_dataset

def msg():
    # randomly select some
    messages = [] 
    dataset = load_dataset('sst', split='test')['sentence']
    labelset = load_dataset('sst', split='test')['label']
    SIZE = 6
    SPLIT = 3
    c = [0] * SPLIT
    for _ in range(SIZE):
        t = random.randrange(len(dataset))
        if labelset[t] < 1/SPLIT and c[0] < 3:
            messages.append({"role": "user", "content": f"\"{dataset[t]}\""})
            messages.append({"role": "assistant", "content": f"{labelset[t]} positive"})
            c[0] += 1
        elif labelset[t] >= 1/SPLIT and labelset[t] < 2/SPLIT and c[1] < 3:
            messages.append({"role": "user", "content": f"\"{dataset[t]}\""})
            messages.append({"role": "assistant", "content": f"{labelset[t]} positive"})
            c[1] += 1
        elif labelset[t] >= 2/SPLIT and c[2] < 3:
            messages.append({"role": "user", "content": f"\"{dataset[t]}\""})
            messages.append({"role": "assistant", "content": f"{labelset[t]} positive"})
            c[2] += 1
    return messages

MSG = msg()
def few_shot_msg(few_shot: bool = False):
    messages = [
    {
        "role": "system", #TODO Add <answer>
        "content": "Please provide an overall positivity prediction value on a scale from 0.00 to 1.00. A value of 1.00 indicates a completely positive movie review, while 0.00 indicates a completely negative movie review. \nThe prediction format should be: \"Prediction Value\" Positive.\nIf uncertain, provide your most accurate estimation."
    }
    ]
    if few_shot:
        messages += MSG
    return messages

def p_e_msg(few_shot: bool = False):
    messages = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).\n\nThen, analyze how important every single word and punctuation token in the review was to your classification. The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word>, <float importance>) for each and every word and punctuation token in the sentence in a format of Python list of tuples. \n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nExample output:\n(<int classification>, <float confidence>)\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]"
    }
    ]
    if few_shot:
        messages += MSG
    return messages

def e_p_msg(few_shot: bool = False):
    messages = [
    {
        "role": "system",
        "content": "You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must analyze how important each and every word and punctuation in the review is in a  Python tuple format: (<word>, <float importance>). The importance should be a decimal number to three decimal places ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>, <float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence.\n\nIt does not matter whether or not the sentence makes sense. Do your best given the sentence.\n\nExample output:\n [(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]\n(<int classification>, <float confidence>)"
    }
    ]
    if few_shot:
        messages += MSG
    return messages