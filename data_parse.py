import json
import pandas as pd

train_data= pd.read_csv("train_patient_descriptions.csv")

# Load the patient data from the attached CSV file and save in JSONL file
train = []
for idx, row in train_data.iterrows():
    id = row['PatientID']
    desc = row['Description']
    label = row["HadHeartAttack"]
    data = {
        "id": id,
        "text": desc,
        "label": label
    }
    train.append(data)

with open('train.jsonl', 'w') as f:
    for data in train:
        json.dump(data, f)
        f.write('\n')

test_data= pd.read_csv("test_patient_descriptions.csv")

# Load the patient data from the attached CSV file and save in JSONL file
test = []
for idx, row in test_data.iterrows():
    id = row['PatientID']
    desc = row['Description']
    label = row["HadHeartAttack"]
    data = {
        "id": id,
        "text": desc,
        "label": label
    }
    test.append(data)

with open('test.jsonl', 'w') as f:
    for data in test:
        json.dump(data, f)
        f.write('\n')