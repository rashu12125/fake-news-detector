import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # For saving models/features

# Download NLTK data (updated for recent NLTK versions - runs automatically if missing)
nltk.download('punkt_tab', quiet=True)  # New: For word_tokenize/sent_tokenize
nltk.download('stopwords', quiet=True)  # English stopwords

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Load the cleaned dataset
df = pd.read_csv('cleaned_fake_news.csv')
print("Loaded cleaned dataset shape:", df.shape)

# Choose text column (use 'text' for full articles; change to 'title' if preferred)
text_column = 'text'  # Or 'title' for headlines only

# Optional: Combine title + text for richer features (uncomment if you want)
# df[text_column] = df['title'].fillna('') + ' ' + df[text_column].fillna('')

print(f"Using column: '{text_column}'")
print(f"Sample before preprocessing: {df[text_column].iloc[0][:200]}...")  # First 200 chars

def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove punctuation, numbers, and special chars (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize into words
    tokens = word_tokenize(text)
    # Remove stopwords and words shorter than 3 chars
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    # Join back into string
    return ' '.join(tokens)

# Apply preprocessing to all texts
print("Preprocessing text... (this may take a few minutes for 38k+ rows)")
df[text_column] = df[text_column].apply(preprocess_text)
print("Text preprocessing completed!")
print(f"Sample after preprocessing: {df[text_column].iloc[0][:200]}...")

# Save preprocessed text dataset (for inspection if needed)
df.to_csv('preprocessed_fake_news.csv', index=False)
print("Preprocessed dataset saved as 'preprocessed_fake_news.csv'")

# Feature Extraction: Convert text to TF-IDF numerical features
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,  # Top 5,000 words/bigrams (adjust for more/less)
    stop_words='english',  # Built-in stopwords (extra safety)
    ngram_range=(1, 2),  # Use single words and 2-word phrases (e.g., "fake news")
    min_df=2  # Ignore words appearing in <2 documents
)
X = vectorizer.fit_transform(df[text_column])  # X: Features (sparse matrix)
y = df['label']  # y: Labels (0=true, 1=fake)

print("TF-IDF features shape:", X.shape)  # e.g., (38646, 5000)
print("Labels shape:", y.shape)
print("Label distribution:\n", y.value_counts())

# Save everything for later use (model training, prediction)
joblib.dump(X, 'tfidf_features.pkl')
joblib.dump(y, 'labels.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Features, labels, and vectorizer saved! Ready for model training.")

# Optional: List files to verify (for debugging)
import os
print("\nFiles created in directory:")
for file in os.listdir('.'):
    if file.endswith(('.pkl', '.csv')):
        print(f"  - {file}")