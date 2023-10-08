import os
import pickle
start = 0
end = 100

prefixed = [filename for filename in os.listdir('.') if filename.startswith("LIME_response_EP_")]
explanations = []
for filename in prefixed:
    with open(filename, 'rb') as handle:
        cur = pickle.load(handle)
    for e in cur:
        explanations.append(e)
with open("LIME_response_EP_%d_%d.pickle" % (start, end), "wb") as handle:
    pickle.dump(explanations, handle, protocol=pickle.HIGHEST_PROTOCOL)
