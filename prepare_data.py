import pandas as pd

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake['label'] = 'FAKE'
real['label'] = 'REAL'

# Combine
df = pd.concat([fake, real])

# Add id column
df['id'] = range(len(df))

# Keep required columns
df = df[['id', 'title', 'text', 'label']]

# Save final dataset
df.to_csv("news.csv", index=False)

print("✅ Dataset ready as news.csv")