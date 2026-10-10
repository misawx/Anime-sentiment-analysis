import joblib
from preprocessing import load_and_clean_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
  # 1. Load and clean data
  df = load_and_clean_data()

  # 2. Define labels (y) based on score median
  median_score = df["score"].median()
  y = (df["score"] > median_score).astype(int)
  print(f"Generated binary labels using 'score' median: {median_score}")

  # 3. TF-IDF Vectorization
  print("Initializing and fitting TF-IDF Vectorizer...")
  tfidf = TfidfVectorizer(max_features=5000)
  X = tfidf.fit_transform(df["cleaned_text"])

  # 4. Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # 5. Train Model
  print("Training Logistic Regression model...")
  model = LogisticRegression(max_iter=1000)
  model.fit(X_train, y_train)

  # 6. Evaluate
  y_pred = model.predict(X_test)
  print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))

  # 7. Save Model & Vectorizer Artifacts for Streamlit
  print("\nSaving model and vectorizer artifacts...")
  joblib.dump(model, "anime_sentiment_model.pkl")
  joblib.dump(tfidf, "tfidf_vectorizer.pkl")
  print("Artifacts saved successfully as .pkl files!")


if __name__ == "__main__":
  main()