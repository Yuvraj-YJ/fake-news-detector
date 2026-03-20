import pandas as pd


print("Running preprocess script...")

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")


fake["label"] = 0
real["label"] = 1


data = pd.concat([fake, real])


print("Dataset shape:", data.shape)
print("\nFirst 5 rows:\n")
print(data.head())