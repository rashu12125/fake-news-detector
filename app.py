import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (runs once, cached)
@st.cache_resource
def download_nltk():
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk()

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load model and vectorizer (cached for speed - loads once)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Fixed: joblib (not joblit)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Run 'python train_model.py' first to train the model.")
        st.stop()

model, vectorizer = load_model()

def preprocess_text(text):
    """Preprocess text using the same method as training for consistency"""
    if not text:
        return ""
    # Lowercase and remove non-letters
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and filter stopwords/short words
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Streamlit App Layout
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detector")
st.markdown("### Powered by Machine Learning (95%+ Accuracy on 38k+ Articles)")
st.markdown("Enter news text (headline or full article) below for instant analysis. Always verify with trusted sources!")

# Sidebar for Instructions and Info
st.sidebar.header("📋 How to Use")
st.sidebar.markdown("""
- **Step 1:** Paste a news headline or article snippet.
- **Step 2:** Click **🔍 Detect Fake News**.
- **Step 3:** View verdict, confidence score, and tips.
- **Model Details:** Trained on ISOT Fake News Dataset using TF-IDF + Logistic Regression.
- **Limitations:** Best for English news; not 100% accurate—use as a tool, not gospel!
""")

st.sidebar.header("🧪 Quick Tests")
st.sidebar.info("Try the examples in the expander below for demos.")

# Main Content Area
col1, col2 = st.columns([3, 1])  # Split layout: Input on left, info on right

with col1:
    # Input Text Area
    user_input = st.text_area(
        "Enter News Text Here:",
        placeholder="Example: 'Trump wins election in landslide, Democrats in total chaos!'",
        height=200,
        help="Paste headline or article. Longer text = better accuracy."
    )

with col2:
    st.markdown("### 💡 Tips")
    st.markdown("- Use full sentences.")
    st.markdown("- Fake news often has sensational words (e.g., 'shocking', 'chaos').")
    st.markdown("- True news: Factual, neutral tone.")

# Predict Button (Centered)
if st.button("🔍 Detect Fake News", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter some news text to analyze!")
    else:
        with st.spinner("Analyzing... (This may take a few seconds on first run)"):
            # Preprocess the input
            processed_text = preprocess_text(user_input)
            
            # Show processed preview
            with st.expander("View Processed Text (Optional)"):
                st.write(f"**Original:** {user_input[:300]}...")
                st.write(f"**Cleaned:** {processed_text[:300]}...")
            
            # Transform to TF-IDF features (same as training)
            features = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(features)[0]  # 0 = True, 1 = Fake
            probability = model.predict_proba(features)[0]  # [True_prob, Fake_prob]
            true_prob = probability[0] * 100
            fake_prob = probability[1] * 100
            confidence = max(true_prob, fake_prob)
            conf_label = "True" if true_prob > fake_prob else "Fake"
            
            # Verdict Display (Large and Visual)
            verdict = "✅ **True News**" if prediction == 0 else "❌ **Fake News**"
            st.markdown(f"# {verdict}")
            
            # Confidence Metrics
            st.metric("Overall Confidence", f"{confidence:.1f}%", f"for {conf_label}")
            
            # Visual Progress Bar
            progress_color = "green" if prediction == 0 else "red"
            st.progress(confidence / 100, text=f"{conf_label} Confidence: {confidence:.1f}%")
            
            # Probability Breakdown (Columns)
            col_true, col_fake = st.columns(2)
            with col_true:
                st.metric("True News Probability", f"{true_prob:.1f}%")
            with col_fake:
                st.metric("Fake News Probability", f"{fake_prob:.1f}%")
            
            # Explanation and Tips
            st.subheader("🔍 Analysis")
            if fake_prob > 70:
                st.error("**High Fake Risk!** 🚨\nThis text shows patterns of sensationalism or misinformation. Cross-check with fact-checkers like Snopes or Reuters.")
            elif true_prob > 70:
                st.success("**Likely Genuine!** 👍\nNeutral and factual tone detected. Still, verify sources for important news.")
            else:
                st.warning("**Uncertain Verdict.** 🤔\nConfidence is medium—try adding more context or check manually.")
            
            # Source Recommendation
            st.info("**Pro Tip:** Use tools like Google Fact Check Explorer or NewsGuard for deeper verification.")

# Test Examples Expander (Below Button)
with st.expander("🧪 Test Examples (Click to Load and Predict)"):
    st.markdown("**Select an example to auto-fill the input above and predict:**")
    examples = {
        "1. Likely Fake (Sensational)": "Trump wins election in landslide, Democrats in total chaos and panic!",
        "2. Likely True (Factual)": "Apple releases new iPhone 15 with advanced camera and battery life improvements.",
        "3. Uncertain (Conspiracy)": "Breaking: UFO sighted over White House, government cover-up confirmed by insiders.",
        "4. Fake (Clickbait)": "Shocking: Celebrities exposed in Hollywood scandal that will ruin lives forever!"
    }
    for label, text in examples.items():
        if st.button(f"Load & Predict: {label}", key=label):
            st.session_state.user_input = text  # Auto-fill (requires rerun)
            st.rerun()  # Refresh page to update input and predict
            st.text_area("Input Auto-Filled:", text, key="auto_input", disabled=True)

# Footer
st.markdown("---")
st.markdown(
    """
    **Built with Streamlit & Scikit-Learn** | Dataset: ISOT Fake News | 
    Created for educational purposes. Not a substitute for professional fact-checking. 
    Questions? Check the sidebar or contact the developer.
    """
)

# Auto-fill handling (if using session state for examples)
if "user_input" not in st.session_state:
    st.session_state.user_input = ""