import os
import pickle
from datasets import load_dataset


dataset = load_dataset("sst")["test"]
dataset = dataset.shuffle(seed=8)['sentence']

dataset = dataset[:100:]
filename = "LIME_explanations_EP.pickle"
with open(filename, "rb") as handle:
    expl = pickle.load(handle)
d = {}
for i in expl:
    d[dataset.index(" ".join(i[1]))] = i
print(type(d.values()))

# for i in d.values():
#     temp_expl = []
#     temp_val = []
#     for j in i:
#         temp_expl.append(j[1][1])
#         temp_val.append(j[1][0])
#     new_expl.append(temp_expl)
#     val.append(temp_val)

r = []
for key, value in d.values():
    # print(key)
    # print(value)
    r.append((key, value))
print(len(r))
with open("LIME_explanations_EP_try.pickle", "wb") as handle:
    pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(r)


# print("****")
# print(val)


# files = ["LIME_response_EP_50_55.pickle", "LIME_response_EP_55_60.pickle", 
#           "LIME_response_EP_60_65.pickle", "LIME_response_EP_65_100.pickle",
#           "LIME_response_EP_0_50.pickle"]
# explanations = []
# sentences = []
# sentences1 = []
# for filename in files:
#     with open(filename, 'rb') as handle:
#         cur = pickle.load(handle)
#     for e in cur:
#         # sentences.append())
#         explanations.append(e)


# # exit()
# with open("LIME_explanations_EP.pickle", "wb") as handle:
#     pickle.dump(explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open("LIME_explanations_PE.pickle", 'rb') as handle:
#     explanations = pickle.load(handle)

# print(len(explanations))