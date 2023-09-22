from datasets import load_dataset
from prediction import Prediction
from messages import *
import csv


dataset = load_dataset('sst', split='test')['sentence']
# openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
  csv_rowlist = []
  prompt = few_shot_msg()
  for i in range(100):
    print(dataset[i])
    instance = Prediction(dataset[i], prompt=prompt)
    result = instance.get_prediction()
    csv_rowlist.append([dataset[i], result])

  with open("gpt_prediction1.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerows(csv_rowlist)