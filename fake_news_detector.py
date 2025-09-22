import os
import pandas as pd

print("Current working directory:", os.getcwd())

# Load the true news CSV
true_df = pd.read_csv('true.csv')

# Load the false news CSV
false_df = pd.read_csv('false.csv')

# Add label columns: 0 for true news, 1 for fake news
true_df['label'] = 0
false_df['label'] = 1

# Combine the two datasets
df = pd.concat([true_df, false_df], ignore_index=True)

# Shuffle the combined dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display label distribution
print("\nLabel distribution:")
print(df['label'].value_counts())

# Save the combined dataset for future use
df.to_csv('combined_fake_news.csv', index=False)
print("\nCombined dataset saved as 'combined_fake_news.csv'")