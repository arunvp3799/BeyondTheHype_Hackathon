import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


# Load the data
inputs = pd.read_csv('inputs.csv')
labels = pd.read_csv('labels.csv')

# Merge on PatientID
data = pd.merge(inputs, labels, on='PatientID')

# Check for missing values and handle them if necessary
data = data.dropna().reset_index(drop=True)

# Split into train and evaluation sets
train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HadHeartAttack'])

# decided to use entire prompt and format in concise way
def create_prompt(row):
    # Example: Convert features into a text description
    prompt = f"The patient is a {row['Sex'].lower()} aged {row['AgeCategory']}. "
    prompt += f"They have a BMI of {row['BMI']:.1f}. "
    prompt += f"Their general health is reported as {row['GeneralHealth'].lower()}. "

    # List key medical history features (only if they have it)
    conditions = []
    if row['HadAngina'] == 1:
        conditions.append('angina')
    if row['HadStroke'] == 1:
        conditions.append('stroke')
    if row['HadAsthma'] == 1:
        conditions.append('asthma')
    if row['HadSkinCancer'] == 1:
        conditions.append('skin cancer')
    if row['HadCOPD'] == 1:
        conditions.append('COPD')
    if row['HadDepressiveDisorder'] == 1:
        conditions.append('depressive disorder')
    if row['HadKidneyDisease'] == 1:
        conditions.append('kidney disease')
    if row['HadArthritis'] == 1:
        conditions.append('arthritis')
    if row['HadDiabetes'] == 'Yes':
        conditions.append('diabetes')
    if row['CovidPos'] == 1:
        conditions.append('COVID-19')

    if conditions:
        prompt += "They have a history of " + ", ".join(conditions) + ". "
    else:
        prompt += "They have no significant medical history. "

    prompt += "Based on this information, is the patient at risk of a heart attack?"

    return prompt

# create dataset class
class HeartAttackDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Preprocess data
        self.texts = []
        self.labels = []
        for _, row in self.dataframe.iterrows():
            prompt = create_prompt(row)
            label = row['HadHeartAttack']
            self.texts.append(prompt)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        prompt = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )

        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = label

        return item

# using distilbert-base-uncased
# this model is smaller with 66M parameters
# it also is classification based
model_name = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# load onto GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# create train and val set
train_dataset = HeartAttackDataset(train_data, tokenizer)
eval_dataset = HeartAttackDataset(eval_data, tokenizer)

# metrics: accuracy, AUC-ROC , precision, recall
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    roc_auc = roc_auc_score(labels, probs[:, 1])
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model='roc_auc',
    greater_is_better=True,
    fp16=True,  # Enable if you have a GPU that supports it
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()