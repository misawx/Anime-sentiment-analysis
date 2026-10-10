import json
import re
import pandas as pd
import spacy

# Load spaCy global model
nlp = spacy.load("en_core_web_sm")


def load_and_clean_data(file_path="data/anime_all_time_20251022_data.json"):
  """Loads the nested Reddit JSON file, extracts submissions,

  and runs the text cleaning pipeline.
  """
  print(f"Loading data from {file_path}...")
  with open(file_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

  # Extract submissions list
  if isinstance(raw_data, dict) and "submissions" in raw_data:
    posts = raw_data["submissions"]
  elif isinstance(raw_data, list):
    posts = []
    for block in raw_data:
      if "submissions" in block:
        posts.extend(block["submissions"])
  else:
    posts = raw_data

  df = pd.DataFrame(posts)
  print(f"Successfully loaded {len(df)} individual Reddit posts!")

  # Preprocessing function
  def preprocess_text(text):
    if not isinstance(text, str):
      return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", "", text)
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)

  print("Cleaning text column (this may take a couple of minutes)...")
  df["cleaned_text"] = df["full_text"].apply(preprocess_text)
  return df