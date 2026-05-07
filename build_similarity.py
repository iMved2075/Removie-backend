import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [item.strip() for item in value.split(",") if item.strip()]
    return []


def split_words(value):
    if isinstance(value, str):
        return value.split()
    return []


def build_similarity(output_path, csv_path=None):
    output_path = Path(output_path)
    if csv_path is None:
        default_csv = output_path.parent / "cleaned_movies.csv"
        alt_csv = output_path.parent.parent / "data" / "cleaned_movies.csv"
        csv_path = default_csv if default_csv.exists() else alt_csv
    csv_path = Path(csv_path)

    movies_df = pd.read_csv(csv_path)
    new_df = movies_df.copy()

    list_columns = [
        "genres",
        "keywords",
        "production_companies",
        "production_countries",
        "cast",
    ]
    for column in list_columns:
        if column in new_df.columns:
            new_df[column] = new_df[column].apply(parse_list)
        else:
            new_df[column] = [[]] * len(new_df)

    if "director" in new_df.columns:
        new_df["director"] = new_df["director"].apply(split_words)
    else:
        new_df["director"] = [[]] * len(new_df)

    for column in ["title", "overview", "tagline"]:
        if column in new_df.columns:
            new_df[column] = new_df[column].apply(split_words)
        else:
            new_df[column] = [[]] * len(new_df)

    new_df["metadata"] = (
        new_df["title"]
        + new_df["genres"]
        + new_df["overview"]
        + new_df["tagline"]
        + new_df["cast"]
        + new_df["director"]
        + new_df["keywords"]
        + new_df["production_companies"]
        + new_df["production_countries"]
    )
    new_df["metadata"] = new_df["metadata"].apply(
        lambda value: " ".join(value) if isinstance(value, list) else ""
    )

    ps = PorterStemmer()

    def stem(text):
        return " ".join(ps.stem(word) for word in text.split())

    new_df["metadata"] = new_df["metadata"].apply(stem)

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(new_df["metadata"]).toarray()
    similarity = cosine_similarity(vectors).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(similarity, output_path, compress=5)


if __name__ == "__main__":
    build_similarity(output_path=Path(__file__).resolve().parent / "similarity.pkl")
