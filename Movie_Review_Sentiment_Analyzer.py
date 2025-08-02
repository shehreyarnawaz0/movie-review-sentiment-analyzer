# 📌 Movie Review Sentiment Analyzer (Using Custom CSV)
# 👨‍💻 Author: Shahereyar
# ✅ This script loads your dataset, trains a sentiment analysis model, and evaluates it.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# ✅ Step 1: Load your dataset
# Make sure your CSV file has two columns: 'review' and 'sentiment' (values: 'positive'/'negative')
df = pd.read_csv("movie_reviews.csv")  # Replace with the actual path if needed

# ✅ Step 2: Display basic info
print("📊 Dataset loaded successfully!")
print(df.head())

# ✅ Step 3: Split into training and testing sets
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ✅ Step 4: Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Step 5: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ✅ Step 6: Predict and evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy on test set: {accuracy * 100:.2f}%")

# ✅ Step 7: Show sample predictions
print("\n📌 Sample Predictions:")
for i in range(3):
    print(f"Review: {X_test.iloc[i][:60]}...")
    print(f"Prediction: {y_pred[i]}")
    print(f"Actual: {y_test.iloc[i]}\n")

# ✅ Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Oranges")
plt.title("Confusion Matrix")
plt.show()

# ✅ Step 9: Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("💾 Model and vectorizer saved successfully.")
