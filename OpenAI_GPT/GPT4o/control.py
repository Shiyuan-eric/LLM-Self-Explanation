import pickle
import os
def loadData(filename):
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
def storeData(filename, data):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

pickle_content = loadData('gpt4o_p_occlusion.pickle')
print(pickle_content[6:7])
# storeData('test_p_occlusion.pickle',pickle_content[6:7])

s = "( A ) superbly controlled , passionate adaptation of Graham Greene 's 1955 novel . -> ( A ) controlled , passionate adaptation of Graham Greene 's 1955 novel ."

s_ep_occ_expl = ([('1955', (0.0005675107406206719, 12)), (')', (0.00039882354222426877, 2)), ('of', (0.00024391087688713199, 8)), ('superbly', (-0.34917515502285174, 3)), ('passionate', (-0.34950642213353467, 6)), ('(', (-0.399985958299836, 0)), ('.', (-0.4490324597497098, 14)), ('Greene', (-0.4491295287678913, 10)), ('adaptation', (-0.44913239722450715, 7)), ('A', (-0.449280295313596, 1)), ('controlled', (-0.4493318467987681, 4)), ('Graham', (-0.44967479563725254, 9)), ('novel', (-0.4497613840713847, 13)), ("'s", (-0.44980893290849755, 11)), (',', (-0.44999885718068555, 5))], ['(', 'A', ')', 'superbly', 'controlled', ',', 'passionate', 'adaptation', 'of', 'Graham', 'Greene', "'s", '1955', 'novel', '.'])
s_ep_self_expl = ([('superbly', (0.9008248449771482, 3)), ('passionate', (0.8004935778664654, 6)), ('controlled', (0.7006681532012318, 4)), ('adaptation', (0.6008676027754928, 7)), ('A', (0.20071970468640396, 1)), ('.', (0.0009675402502901433, 14)), ('Greene', (0.0008704712321086546, 10)), ('1955', (0.0005675107406206719, 12)), (')', (0.00039882354222426877, 2)), ('Graham', (0.00032520436274739007, 9)), ('of', (0.00024391087688713199, 8)), ('novel', (0.0002386159286152202, 13)), ("'s", (0.00019106709150239055, 11)), ('(', (1.4041700164018957e-05, 0)), (',', (1.1428193144282783e-06, 5))], ['(', 'A', ')', 'superbly', 'controlled', ',', 'passionate', 'adaptation', 'of', 'Graham', 'Greene', "'s", '1955', 'novel', '.'])
