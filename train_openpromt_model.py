from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from transformers import AdamW

import pandas as pd
import torch

# config
bach_size = 10
model_type = 't5'
model_name = 't5-large'

# Define the base file path
base_path = '/scratch/yt3182/hackathon/'
# Load the train&test dataset
train_data_file_path = base_path + 'code/data/train_patient_descriptions.csv'
train_data = pd.read_csv(train_data_file_path)

test_data_file_path = base_path + 'code/data/test_patient_descriptions.csv'
test_data = pd.read_csv(test_data_file_path)

# Preprocess data
def preprocess_data(data):
    processed_data = []
    for index, row in data.iterrows():
        data_point = {
            'id': row['PatientID'],
            'Description': row['Description'],
            'label': 1 if row['HadHeartAttack'] == 'yes' else 0,
            'HadHeartAttack': row['HadHeartAttack']
        }
        processed_data.append(data_point)
    return processed_data

#train
train = preprocess_data(train_data)
#test
test = preprocess_data(test_data)


# in the answer 0 mean no.
verbliz = {"0": ["no","low"], "1":["yes","high"]}
label_list = ["no","yes"]
label_dict = {"no": 0, "yes":1}
label2answer_dict = {0: "no", 1: "yes"}
label_words = {label_dict[k]: [k.split(".")[-1]] for k in label_dict}
train = [{'input': k['Description'] , 'answer': k['HadHeartAttack'], 'label':k['label']} for k in train]
train = [{'input': k['Description'] , 'answer': k['HadHeartAttack'], 'label':k['label']} for k in test]

# create data set
dataset = {}
dataset['train'] = []
label_amount_dict = {label_dict[k]: 0 for k in label_list}
print(label_dict)
for index, row in enumerate(train):
    input_example = InputExample(text_a=row['input'], text_b='', label=label_dict[row['label']], guid=index)
    label_amount_dict[label_dict[row['label']]] += 1
    dataset['train'].append(input_example)
# weight loss vector
weight_vector = [1-(lexical_amount_dict[k]/len(train)) for k in lexical_amount_dict]
class_weights=torch.FloatTensor(weight_vector).cuda()


dataset['test'] = []
for index, row in enumerate(test):
    input_example = InputExample(text_a=row['input'], text_b='', label=label_dict[row['label']], guid=index)
    dataset['test'].append(input_example)



# load model
plm, tokenizer, model_config, WrapperClass = load_plm(model_type, model_name)


# create template
mytemplate = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} The probility of get heart attack is  {"mask"}.')
wrapped_t5tokenizer = WrapperClass(max_seq_length=256, decoder_max_length=5, tokenizer=tokenizer,
                                   truncate_method="head")
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=5,
                                    batch_size=bach_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head", clean=False)
myverbalizer = ManualVerbalizer(tokenizer, num_classes=2, classes=[label_dict[k] for k in label_dict],
                                label_words=label_words)
use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

# ## below is standard training
loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
#loss_func =  torch.nn.CrossEntropyLossFlat(weight=class_weights)
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
acc_list = []
max_acc = 0
for epoch in range(30):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

        if step % 200 == 1 and step!=1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
            prompt_model.eval()
            validation_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                                     tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                                     decoder_max_length=10,
                                                     batch_size=1, shuffle=False, teacher_forcing=False,
                                                     predict_eos_token=False,
                                                     truncate_method="head")

            allpreds = []
            alllabels = []
            for step, inputs in enumerate(validation_dataloader):
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
            print("validation acc:", acc)
            acc_list.append(acc)
            if acc > max_acc:
                max_acc = acc
                torch.save(prompt_model.state_dict(), base_path + 'models/'+ model_name +'1.model')
                write_file_to_path([acc], base_path + 'models/'+ model_name +'1.json')
            prompt_model.train()
    
validation_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate,
                                         tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                         decoder_max_length=5,
                                         batch_size=1, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="head")

allpreds = []
alllabels = []
for step, inputs in enumerate(validation_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
print("test acc:", acc)
