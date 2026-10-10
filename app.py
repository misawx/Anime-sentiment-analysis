import joblib
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Anime Sentiment Classifier", page_icon="🎌", layout="centered"
)

# Load saved model and vectorizer
@st.cache_resource
def load_artifacts():
  model = joblib.load("anime_sentiment_model.pkl")
  tfidf = joblib.load("tfidf_vectorizer.pkl")
  return model, tfidf


model, tfidf = load_artifacts()

# App UI Layout
st.title("🎌 Reddit Anime Sentiment Analyzer")
st.write(
    "Enter an anime review, thought, or discussion snippet below. Our machine"
    " learning model will predict whether it leans **Positive** (high engagement"
    " score) or **Neutral/Negative** (lower engagement score)."
)

# User input text box
user_input = st.text_area(
    "Enter your anime text here:",
    placeholder=(
        "e.g., The animation quality in this season was absolute masterpiece"
        " level..."
    ),
)

if st.button("Predict Sentiment"):
  if user_input.strip() == "":
    st.warning("Please enter some text before analyzing.")
  else:
    # Transform input using the saved TF-IDF vectorizer
    input_vectorized = tfidf.transform([user_input])

    # Predict using the saved Logistic Regression model
    prediction = model.predict(input_vectorized)[0]
    prediction_proba = model.predict_proba(input_vectorized)[0]

    # Display results nicely
    st.subheader("Prediction Result:")
    if prediction == 1:
      st.success(
          f"🔥 **Positive Sentiment / High Score Potential** (Confidence:"
          f" {prediction_proba[1]*100:.1f}%)"
      )
    else:
      st.info(
          f"📉 **Neutral / Lower Score Potential** (Confidence:"
          f" {prediction_proba[0]*100:.1f}%)"
      )