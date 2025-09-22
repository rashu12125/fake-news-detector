import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the saved features, labels, and vectorizer
print("Loading saved components...")
X = joblib.load('tfidf_features.pkl')  # Features
y = joblib.load('labels.pkl')          # Labels
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # For future predictions
print("Loaded features shape:", X.shape)
print("Labels shape:", y.shape)
print("Label distribution:\n", y.value_counts())

# Split data: 80% train, 20% test (stratified to keep balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Train the model: Logistic Regression (simple and effective for text)
print("\nTraining model...")
model = LogisticRegression(max_iter=1000, random_state=42)  # max_iter prevents convergence warnings
model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f} (expect 0.95+ for this dataset)")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['True News (0)', 'Fake News (1)']))

print("\nConfusion Matrix (rows: actual, cols: predicted):")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'fake_news_model.pkl')
print("\nModel saved as 'fake_news_model.pkl' - Ready for predictions!")