import pandas as pd
from sklearn.model_selection import train_test_split

base_path = '/scratch/yt3182/hackathon/'

# Load the patient data from the attached CSV file
inputs_file_path = base_path + 'ha_train_set/inputs.csv'
data = pd.read_csv(inputs_file_path)

# Load the labeled data containing heart disease results
labeled_data_file = base_path + 'ha_train_set/labels.csv'
labeled_data = pd.read_csv(labeled_data_file)

# Function to generate a natural language description of the patient
def generate_patient_description(patient):
    # Extracting data from patient dictionary
    sex = patient['Sex']
    bmi = round(patient['BMI'], 2)
    height = round(patient['HeightInMeters'], 2)
    weight = round(patient['WeightInKilograms'], 2)
    had_angina = 'had angina' if patient['HadAngina'] == 1 else 'did not have angina'
    had_stroke = 'had a stroke' if patient['HadStroke'] == 1 else 'did not have a stroke'
    had_copd = 'had COPD' if patient['HadCOPD'] == 1 else 'did not have COPD'
    had_kidney_disease = 'had kidney disease' if patient['HadKidneyDisease'] == 1 else 'did not have kidney disease'
    had_depressive_disorder = 'had a depressive disorder' if patient['HadDepressiveDisorder'] == 1 else 'did not have a depressive disorder'
    alcohol = 'drinks alcohol' if patient['AlcoholDrinkers'] == 1 else 'does not drink alcohol'
    chest_scan = 'had a chest scan' if patient['ChestScan'] == 1 else 'did not have a chest scan'
    high_risk_last_year = 'was considered high risk last year' if patient['HighRiskLastYear'] == 1 else 'was not considered high risk last year'
    difficulty_walking = 'has difficulty walking' if patient['DifficultyWalking'] == 1 else 'does not have difficulty walking'
    difficulty_errands = 'has difficulty completing errands alone' if patient['DifficultyErrands'] == 1 else 'does not have difficulty completing errands alone'
    had_arthritis = 'had arthritis' if patient['HadArthritis'] == 1 else 'did not have arthritis'
    hiv_testing = 'has been tested for HIV' if patient['HIVTesting'] == 1 else 'has not been tested for HIV'
    had_asthma = 'had asthma' if patient['HadAsthma'] == 1 else 'did not have asthma'
    difficulty_concentrating = 'has difficulty concentrating' if patient['DifficultyConcentrating'] == 1 else 'does not have difficulty concentrating'
    had_skin_cancer = 'had skin cancer' if patient['HadSkinCancer'] == 1 else 'did not have skin cancer'
    blind_or_vision_difficulty = 'is blind or has vision difficulties' if patient['BlindOrVisionDifficulty'] == 1 else 'does not have vision difficulties'
    difficulty_dressing_bathing = 'has difficulty dressing or bathing' if patient['DifficultyDressingBathing'] == 1 else 'does not have difficulty dressing or bathing'
    covid_pos = 'tested positive for COVID-19' if patient['CovidPos'] == 1 else 'did not test positive for COVID-19'
    pneumo_vax_ever = 'has received the pneumococcal vaccine' if patient['PneumoVaxEver'] == 1 else 'has not received the pneumococcal vaccine'
    deaf_or_hard_of_hearing = 'is deaf or hard of hearing' if patient['DeafOrHardOfHearing'] == 1 else 'does not have hearing difficulties'
    flu_vax_last_12 = 'received a flu vaccine in the last 12 months' if patient['FluVaxLast12'] == 1 else 'did not receive a flu vaccine in the last 12 months'

    pronoun = 'he' if sex.lower() == 'male' else 'she'

    # Constructing the description
    description = (
        f"The patient is a {sex.lower()} with a BMI of {bmi:.2f}, weighing {weight:.2f} kg and standing {height:.2f} meters tall. "
        f"The patient {had_angina}, {had_stroke}, {had_copd}, {had_kidney_disease}, and {had_depressive_disorder}. "
        f"{pronoun.capitalize()} {difficulty_walking} and {difficulty_errands}. "
        f"The patient {alcohol}. {pronoun.capitalize()} {chest_scan} and {high_risk_last_year}. "
        f"The patient {had_arthritis}, {hiv_testing}, {had_asthma}, and {difficulty_concentrating}. "
        f"The patient {had_skin_cancer}, {blind_or_vision_difficulty}, and {difficulty_dressing_bathing}. "
        f"The patient {covid_pos}, {pneumo_vax_ever}, and {deaf_or_hard_of_hearing}. "
        f"Additionally, the patient {flu_vax_last_12}."
    )

    # Adding a question for prediction
    description += " Does this patient have heart disease?"

    return description

# Generate descriptions for all patients
descriptions = []
for index, patient in data.iterrows():
    patient_id = patient['PatientID']
    description = generate_patient_description(patient)
    descriptions.append({'PatientID': patient_id, 'Description': description})

# Create a DataFrame with the generated descriptions
description_df = pd.DataFrame(descriptions)

# Merge the description data with labeled heart disease data
merged_data = pd.merge(description_df, labeled_data, on='PatientID')

# Update the heart disease column to "yes" or "no"
merged_data['HadHeartAttack'] = merged_data['HadHeartAttack'].apply(lambda x: 'yes' if x == 1 else 'no')

# Split the data into training and test sets
heart_attack_yes = merged_data[merged_data['HadHeartAttack'] == 'yes']
heart_attack_no = merged_data[merged_data['HadHeartAttack'] == 'no']

train_yes = heart_attack_yes.sample(frac=0.8, random_state=42)
test_yes = heart_attack_yes.drop(train_yes.index)
train_no = heart_attack_no.sample(frac=0.8, random_state=42)
test_no = heart_attack_no.drop(train_no.index)

train_data = pd.concat([train_yes, train_no]).sample(frac=1, random_state=42)  # Shuffle the training data
test_data = pd.concat([test_yes, test_no]).sample(frac=1, random_state=42)    # Shuffle the test data

# Output the train and test data to CSV files
train_output_file_path = base_path + 'code/data/train_patient_descriptions.csv'
test_output_file_path = base_path + 'code/data/test_patient_descriptions.csv'
train_data.to_csv(train_output_file_path, index=False)
test_data.to_csv(test_output_file_path, index=False)

print(f"Training data has been saved to {train_output_file_path}")
print(f"Test data has been saved to {test_output_file_path}")
