import pandas as pd  # This imports pandas and aliases it as 'pd' (fixes the NameError)

# Load the combined dataset
df = pd.read_csv('combined_fake_news.csv')

# Explore data
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nMissing values per column:")
print(df.isnull().sum())

# Assume the text column is named 'text' (common in fake news datasets)
# If your column is different (e.g., 'title' or 'statement'), replace 'text' below
# We'll check this automatically - run and see the column names output
text_column = 'text'  # Update this after seeing the column names (e.g., if it's 'title', change to 'title')

# Basic cleaning: Remove duplicates and drop rows with missing text
# First, check if text_column exists
if text_column not in df.columns:
    print(f"\nWarning: '{text_column}' column not found. Available columns: {df.columns.tolist()}")
    print("Please update 'text_column' in the code and re-run.")
else:
    df = df.dropna(subset=[text_column])
    df = df.drop_duplicates(subset=[text_column])
    print(f"\nAfter cleaning: {df.shape[0]} rows")

    # Display sample text
    print("\nSample true news (label 0):")
    print(df[df['label'] == 0][text_column].iloc[0])
    print("\nSample fake news (label 1):")
    print(df[df['label'] == 1][text_column].iloc[0])

    # Save the cleaned dataset for future steps
    df.to_csv('cleaned_fake_news.csv', index=False)
    print("\nCleaned dataset saved as 'cleaned_fake_news.csv'")