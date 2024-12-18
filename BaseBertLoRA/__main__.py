import argparse
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import PeftModel if you used LoRA
from peft import PeftModel

PREDICTION_WINDOW_MONTHS = [3, 6, 9, 12]  # Constant for this charge-off prediction task.


def main(test_set_dir: str, results_dir: str):
    
    # Load test set data.
    input_df = pd.read_csv(os.path.join(test_set_dir, "inputs.csv"))

    # ---------------------------------
    # START PROCESSING TEST SET INPUTS

    # Load the tokenizer and base model
    model_name = 'hsp287/heart_attack_pred_bert_lora'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model
    # load the base model first and then load the PEFT model (LoRA)
    # currently LoRA model is not trained
    base_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_name)

    # Ensure the model is in evaluation mode
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the prompt creation function (same as used during training)
    def create_prompt(row):
        prompt = f"The patient is a {row['Sex'].lower()} aged {row['AgeCategory']}. "
        prompt += f"They have a BMI of {row['BMI']:.1f}. "
        prompt += f"Their general health is reported as {row['GeneralHealth'].lower()}. "

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
        
    predictions = []
    patients = list(input_df.PatientID)

    for idx, row in input_df.iterrows():
        # Create the prompt for the current row
        prompt = create_prompt(row)

        # Tokenize the prompt
        inputs = tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )

        # Move inputs to device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Run the model and get the prediction
        with torch.no_grad():
            output = model(**inputs)

        # Get the predicted label
        logits = output.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

        # Append the prediction to the list
        predictions.append(prediction)

    output_df = pd.DataFrame(columns=["PatientID", "HadHeartAttack"])
    output_df["PatientID"] = patients
    output_df["HadHeartAttack"] = predictions.cpu().numpy()

    # END PROCESSING TEST SET INPUTS
    # ---------------------------------

    # Save the results as "results.csv" in the specified directory
    print(f"Saving to {results_dir}.")
    output_df.to_csv(os.path.join(results_dir, "results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bth_test_set",
        type=str,
        required=True
    )
    parser.add_argument(
        "--bth_results",
        type=str,
        required=True
    )
    args = parser.parse_args()
    main(args.bth_test_set, args.bth_results)