import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("T-MOBILE SENTIMENT ANALYSIS MODEL TRAINING")
print("="*60)

# Load the cleaned data
print("\n📊 Loading cleaned data...")
df = pd.read_csv('tmobile_reddit_cleaned.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nSentiment distribution:\n{df['SentimentLabel'].value_counts()}")

# Prepare features and labels
print("\n🔧 Preparing features and labels...")
X = df['FeedbackText']  # Input: cleaned text
y = df['SentimentLabel']  # Output: Positive/Neutral/Negative

# Split data into training and testing sets
print("\n✂️ Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# Convert text to numerical features using TF-IDF
print("\n🔤 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Feature matrix shape: {X_train_vec.shape}")

# Train multiple models and compare
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\n🤖 Training {name}...")
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"✓ {name} Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model_name} ({results[best_model_name]:.2%} accuracy)")
print("="*60)

# Retrain best model on full training data
print(f"\n🔄 Retraining {best_model_name} on full training data...")
best_model.fit(X_train_vec, y_train)

# Final evaluation
y_pred_final = best_model.predict(X_test_vec)
print("\n📈 Final Model Performance:")
print(classification_report(y_test, y_pred_final))

# Confusion Matrix
print("\n📊 Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, 
            yticklabels=best_model.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")

# Save the model and vectorizer
print("\n💾 Saving model and vectorizer...")
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Model saved as 'sentiment_model.pkl'")

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ Vectorizer saved as 'tfidf_vectorizer.pkl'")

# Test the model with sample predictions
print("\n" + "="*60)
print("TESTING MODEL WITH SAMPLE PREDICTIONS")
print("="*60)

test_samples = [
    "T-Mobile service is amazing! Fast speeds and great coverage.",
    "Customer service was okay, nothing special.",
    "Terrible experience. My phone doesn't work and billing is a nightmare."
]

for i, sample in enumerate(test_samples, 1):
    sample_vec = vectorizer.transform([sample])
    prediction = best_model.predict(sample_vec)[0]
    probabilities = best_model.predict_proba(sample_vec)[0]
    
    print(f"\nSample {i}: {sample[:60]}...")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {max(probabilities):.2%}")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Model saved and ready for deployment")
print("2. Use 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' in Streamlit app")
print("3. Load model to predict sentiment on new customer feedback")