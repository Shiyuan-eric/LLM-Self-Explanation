from datasets import load_dataset
import pickle

def load_file(filename):
    with open(filename, "rb") as handle:
        expl = pickle.load(handle)
    # print(expl)
    prediction = []
    for key, value in expl.items():
        temp = []
        prediction.append(value[1][1] if value[1][0] == 1 else 1-value[1][1])
    return prediction

def evaluate_accuracy(prediction):
    dataset = load_dataset('sst', split='test')
    dataset = dataset.shuffle(seed=8)['label']
    accuracy = []
    for i in range(len(prediction)):
        if (prediction[i] > 0.5 and dataset[i] > 0.5) or (prediction[i] <= 0.5 and dataset[i] <= 0.5):
            accuracy.append(1)
        else:
            accuracy.append(0)
    return accuracy.count(1)/len(accuracy)

PE_prediction = load_file("parse_topk.pickle")
EP_prediction = load_file("parse_topk_EP.pickle")
print(evaluate_accuracy(PE_prediction))
print(evaluate_accuracy(EP_prediction))

