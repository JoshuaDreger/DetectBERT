import pandas as pd
import os
import random

# Load the input dataframe (replace with your file path)

dataset = "transcend"
df = pd.read_csv(f"/work/j.dreger/master-work/jupyter/data/{dataset}_smooth.csv")

# Ensure output directories exist
ouput_folder = f"/work/j.dreger/master-work/repos/detectbert/data/apk_splits/250125_txt/{dataset}/"
os.makedirs(ouput_folder, exist_ok=True)

# Helper function to create file paths
def generate_file_paths(df_subset, file_name):
    file_paths = []
    for _, row in df_subset.iterrows():
        label = "goodware" if row['malware'] == 0 else "malware"
        file_paths.append(f"{label}/{row['sha256']}.png")
    
    with open(file_name, 'w') as f:
        f.write("\n".join(file_paths))

# Split the dataset into 80% train and 20% test with equal label distribution
malware_df = df[df['malware'] == 1]
goodware_df = df[df['malware'] == 0]

# Random splits with balanced label distribution
train_malware = malware_df.sample(frac=0.8, random_state=42)
test_malware = malware_df.drop(train_malware.index)

train_goodware = goodware_df.sample(frac=0.8, random_state=42)
test_goodware = goodware_df.drop(train_goodware.index)

train_random = pd.concat([train_malware, train_goodware]).sample(frac=1, random_state=42)
test_random = pd.concat([test_malware, test_goodware]).sample(frac=1, random_state=42)

# Time-based splits
train_timebased = df[df['split'] == 'train'].sample(frac=1, random_state=42)
test_timebased = df[df['split'] == 'test'].sample(frac=1, random_state=42)
# Generate the .txt files
generate_file_paths(train_random, ouput_folder + "train_random.txt")
generate_file_paths(test_random, ouput_folder + "test_random.txt")
generate_file_paths(train_timebased, ouput_folder + "train_timebased.txt")
generate_file_paths(test_timebased, ouput_folder + "test_timebased.txt")

print(".txt files have been successfully created!")
