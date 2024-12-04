import os
import torch
import transformers
from tqdm import tqdm
import pandas as pd
from trl import SFTTrainer
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

def load_bnb_model():
    os.environ["HF_TOKEN"] = "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"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id = "google/gemma-2b-it"

    model = AutoModelForCausalLM.from_pretrained(model_id, config=bnb_config, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
    return model, tokenizer

def load_data():
    data_files = {
        "train": "data/train.jsonl",
        "test": "data/test.jsonl"
    }
    dataset = load_dataset("json", data_files=data_files)
    return dataset

def generate_prompt(example):

    prefix_text = """Below is a patient report text that contains the information about patient's health confitions, it is followed up by a question. Answer the question as 'yes' or 'no'."""
    text = f"""<start_of_turn>user {prefix_text} \n <text>:\n {example["text"]} \n<end_of_text>\n <end_of_turn>\n<start_of_turn>model {example["label"]} <end_of_turn>"""
    return [text]

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_lora_config(model):

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r = 16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )

    model = get_peft_model(model, lora_config)

    return model, lora_config

def calculate_accuracy(model, tokenizer, dataset):

    true = []
    pred = []
    count = 0
    for sample in tqdm(dataset["test"]):
        if count == 100:
            break
        text = sample["text"]
        label = sample["label"]
        prompt = generate_prompt({"text": text, "label": label})
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to("cuda")
        output = model.generate(input_ids, do_sample=False,  max_new_tokens=4)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred.append(output_text)
        true.append(label)
        count += 1

    out = {}
    out["true"] = true
    out["pred"] = pred

    out_df = pd.DataFrame.from_dict(out)
    out_df.to_csv("results.csv", index=False)

def train():

    print(f"Starting the Process ...")
    model, tokenizer = load_bnb_model()
    print(f"Model Loading Done ...")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = load_data()
    print(f"Dataset Loading Done ...")
    print(f"Example: {dataset['train'][0]}")
    model, lora_config = load_lora_config(model)
    model.to("cuda")
    calculate_accuracy(model, tokenizer, dataset)

    model.train()
    # model.to("cuda")
    model.zero_grad()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        max_seq_length=1024,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=0,
            max_steps=100,
            learning_rate=1e-4,
            logging_steps=1,
            logging_dir="logs",
            output_dir="output",
            optim="paged_adamw_8bit",
            save_strategy="epoch",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        format_function=generate_prompt,
    )

    trainer.train()

if __name__ == "__main__":
    train()