import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data (quietly, if missing)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
print("Loading model and vectorizer...")
try:
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Missing file {e}. Run train_model.py first.")
    exit()

def preprocess_text(text):
    """Same preprocessing as in training - for consistency"""
    if not text:
        return ""
    # Lowercase and clean (remove non-letters)
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and remove stopwords/short words
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

print("\n=== Fake News Detector Ready! ===")
print("Enter news text (headline or article). Type 'quit' to exit.")
print("Predictions: 0 = True News, 1 = Fake News\n")

while True:
    # Get user input
    user_input = input("\nEnter news text: ").strip()
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    if not user_input:
        print("Please enter some text.")
        continue
    
    # Preprocess the input
    processed_text = preprocess_text(user_input)
    print(f"Processed: {processed_text[:100]}...")  # Show first 100 chars
    
    # Convert to TF-IDF features (same as training)
    features = vectorizer.transform([processed_text])  # Single sample
    
    # Predict
    prediction = model.predict(features)[0]  # 0 or 1
    probability = model.predict_proba(features)[0]  # Confidence scores
    true_prob = probability[0] * 100  # % chance true
    fake_prob = probability[1] * 100  # % chance fake
    
    # Output result
    verdict = "True News" if prediction == 0 else "Fake News"
    print(f"\nPrediction: {verdict}")
    print(f"Confidence: {max(true_prob, fake_prob):.1f}% ({'True' if true_prob > fake_prob else 'Fake'})")
    print(f"Full Probabilities: True={true_prob:.1f}%, Fake={fake_prob:.1f}%")
    
    # Optional: Explain (simple - based on probability)
    if fake_prob > 70:
        print("⚠️  High confidence fake - Check sources!")
    elif true_prob > 70:
        print("✅ Likely real - But always verify.")
    else:
        print("🤔 Uncertain - More context needed.")