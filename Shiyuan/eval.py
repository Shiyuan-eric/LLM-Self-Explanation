from datasets import load_dataset
import csv

labels = []
with open('gpt_prediction1.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for lines in csv_reader:
        labels.append(lines[1])

dataset = load_dataset('sst', split='test')['label']
to_sum = []
correct_classification = 0
for i in range(len(labels)):
    to_sum.append(abs(float(labels[i])-float(dataset[i])))
    if (float(labels[i]) >= 0.5 and float(dataset[i]) >= 0.5) or (float(labels[i]) < 0.5 and float(dataset[i]) < 0.5):
        correct_classification += 1

output = sum(to_sum)/(len(labels))
print(output)

correctness = correct_classification / (len(labels))
print("Classification correctness:", correctness)