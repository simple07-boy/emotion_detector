import pandas as pd

# Change these paths if needed
train_df = pd.read_csv("train.txt", names=["Text", "Emotion"], sep=";")
val_df = pd.read_csv("val.txt", names=["Text", "Emotion"], sep=";")

# Combine and save
combined = pd.concat([train_df, val_df], ignore_index=True)
combined.to_csv("emotions_dataset.csv", index=False)

print("Dataset CSV created: emotions_dataset.csv")