from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle
import joblib
import numpy as np
import os
from pathlib import Path
import requests
import time
import ast

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": ["https://removie-frontend.vercel.app"]}},
)

BASE_DIR = Path(__file__).resolve().parent
MOVIES_CSV_PATH = BASE_DIR / "cleaned_movies.csv"
SIMILARITY_PATH = BASE_DIR / "similarity.pkl"

movies_df = pd.read_csv(MOVIES_CSV_PATH)

def ensure_similarity_file():
    if SIMILARITY_PATH.exists():
        return
    try:
        from build_similarity import build_similarity
        build_similarity(output_path=SIMILARITY_PATH, csv_path=MOVIES_CSV_PATH)
    except Exception as exc:
        raise RuntimeError(
            "similarity.pkl is missing. Run 'python build_similarity.py' during deployment."
        ) from exc

ensure_similarity_file()
model = joblib.load(SIMILARITY_PATH)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_CACHE = {}
TMDB_CACHE_TTL_SECONDS = 6 * 60 * 60
TMDB_SESSION = requests.Session()
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 30
RATE_LIMIT_STATE = {}
RESULT_CACHE = {}
RESULT_CACHE_TTL_SECONDS = 60

def load_env_local(env_path: Path):
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = value
    except OSError:
        pass

repo_root = Path(__file__).resolve().parent.parent
load_env_local(repo_root / ".env.local")
load_env_local(Path(__file__).resolve().parent / ".env.local")

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
print(f"TMDB_API_KEY loaded: {'yes' if TMDB_API_KEY else 'no'}")

def dataframe_to_records(df):
    """Convert NaN and infinity values to None for JSON serialization."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object)
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient='records')

def tmdb_get_movie(tmdb_id):
    if not TMDB_API_KEY or not tmdb_id:
        return {}
    cache_entry = TMDB_CACHE.get(tmdb_id)
    if cache_entry and cache_entry["expires_at"] > time.time():
        return cache_entry["data"]
    try:
        tmdb_id = int(tmdb_id)
        response = TMDB_SESSION.get(
            f"{TMDB_API_BASE}/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5,
        )
        if response.ok:
            data = response.json()
            TMDB_CACHE[tmdb_id] = {
                "data": data,
                "expires_at": time.time() + TMDB_CACHE_TTL_SECONDS,
            }
            return data
    except requests.RequestException:
        pass
    TMDB_CACHE[tmdb_id] = {"data": {}, "expires_at": time.time() + 120}
    return {}

def get_client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"

def check_rate_limit():
    now = time.time()
    client = get_client_ip()
    entry = RATE_LIMIT_STATE.get(client)
    if not entry or now >= entry["reset_at"]:
        RATE_LIMIT_STATE[client] = {"count": 1, "reset_at": now + RATE_LIMIT_WINDOW_SECONDS}
        return False, 0
    if entry["count"] >= RATE_LIMIT_MAX_REQUESTS:
        retry_after = max(1, int(entry["reset_at"] - now))
        return True, retry_after
    entry["count"] += 1
    return False, 0

def parse_limit(value, default=30, max_value=50):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < 1:
        return default
    return min(parsed, max_value)

def enrich_records(records, mode="full"):
    if not TMDB_API_KEY:
        return records
    if mode == "none":
        return records
    for record in records:
        tmdb_id = record.get("id")
        if not tmdb_id:
            continue
        data = tmdb_get_movie(tmdb_id)
        if not data:
            continue
        poster_path = data.get("poster_path")
        backdrop_path = data.get("backdrop_path")
        if poster_path:
            record["poster_url"] = f"{TMDB_IMAGE_BASE}/w500{poster_path}"
        if backdrop_path:
            record["backdrop_url"] = f"{TMDB_IMAGE_BASE}/w780{backdrop_path}"

        if mode == "posters":
            continue

        record["release_date"] = data.get("release_date") or record.get("release_date")
        record["vote_average"] = data.get("vote_average") or record.get("vote_average")
        record["vote_count"] = data.get("vote_count") or record.get("vote_count")
        record["runtime"] = data.get("runtime") or record.get("runtime")
        record["tagline"] = data.get("tagline") or record.get("tagline")
        record["language"] = data.get("original_language") or record.get("original_language") or record.get("language")

        genres = data.get("genres")
        if genres:
            record["genres"] = ", ".join(
                [g.get("name") for g in genres if g.get("name")]
            )
    return records

def serialize_movies(df, enrich="none"):
    records = dataframe_to_records(df)
    return enrich_records(records, mode=enrich)

def get_cached_result(key):
    entry = RESULT_CACHE.get(key)
    if not entry:
        return None
    if entry["expires_at"] <= time.time():
        RESULT_CACHE.pop(key, None)
        return None
    return entry["data"]

def set_cached_result(key, data):
    RESULT_CACHE[key] = {
        "data": data,
        "expires_at": time.time() + RESULT_CACHE_TTL_SECONDS,
    }

@app.route('/')
def home():
    return "Welcome to ReMovie!"

@app.route('/api/movies', methods=['GET'])
def get_movies():
    limit = parse_limit(request.args.get('limit'), default=0, max_value=100)
    randomize = request.args.get('random', default=0, type=int)
    enrich = request.args.get('enrich', default='none')
    df = movies_df
    if limit and limit > 0:
        if randomize:
            df = df.sample(n=min(limit, len(df)))
        else:
            df = df.head(limit)
    return jsonify(serialize_movies(df, enrich=enrich))

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json(silent=True) or {}
    movie_title = data.get('movie_title')
    if not movie_title or not isinstance(movie_title, str):
        return jsonify({'error': 'Movie title is required'}), 400

    normalized = movie_title.strip()
    if not normalized:
        return jsonify({'error': 'Movie title is required'}), 400

    exact_match = movies_df[movies_df['title'] == normalized]
    if exact_match.empty:
        lowered = normalized.lower()
        exact_match = movies_df[movies_df['title'].str.lower() == lowered]

    if exact_match.empty:
        return jsonify({'recommended_movies': []})

    movie_index = exact_match.index[0]
    distances = sorted(list(enumerate(model[movie_index])), reverse=True, key=lambda x: x[1])

    recommended_movies = []
    limit = parse_limit(data.get('limit'), default=5, max_value=10)
    for i in distances[1:limit + 1]:
        recommended_movies.append(movies_df.iloc[i[0]]['title'])
    return jsonify({'recommended_movies': recommended_movies})

@app.route('/api/genres', methods=['GET'])
def get_genres():
    genres = set()
    for genre_list in movies_df['genres']:
        if not genre_list:
            continue
        if isinstance(genre_list, list):
            parsed = genre_list
        elif isinstance(genre_list, str):
            try:
                parsed = ast.literal_eval(genre_list)
            except (ValueError, SyntaxError):
                parsed = [g.strip() for g in genre_list.split(',')]
        else:
            parsed = []
        for genre in parsed:
            if not genre:
                continue
            cleaned = str(genre).strip().strip("'").strip('"')
            if cleaned:
                genres.add(cleaned)
    return jsonify(sorted(genres))

@app.route('/api/movies/genre', methods=['POST'])
def get_movies_by_genre():
    data = request.get_json(silent=True) or {}
    genre = data.get('genre')
    if not genre or not isinstance(genre, str) or not genre.strip():
        return jsonify({'error': 'Genre is required'}), 400
    limit = parse_limit(request.args.get('limit'), default=30)
    enrich = request.args.get('enrich', default='posters')
    cache_key = f"genre:{genre.lower().strip()}:{limit}:{enrich}"
    cached = get_cached_result(cache_key)
    if cached is not None:
        return jsonify(cached)
    filtered_movies = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)]
    if limit:
        filtered_movies = filtered_movies.head(limit)
    data = serialize_movies(filtered_movies, enrich=enrich)
    set_cached_result(cache_key, data)
    return jsonify(data)

@app.route('/api/movies/search', methods=['POST'])
def search_movies():
    limited, retry_after = check_rate_limit()
    if limited:
        response = jsonify({'error': 'Rate limit exceeded', 'retry_after': retry_after})
        response.headers['Retry-After'] = str(retry_after)
        return response, 429

    data = request.get_json(silent=True) or {}
    query = data.get('query')
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({'error': 'Query is required'}), 400
    limit = parse_limit(request.args.get('limit'), default=30)
    enrich = request.args.get('enrich', default='posters')
    cache_key = f"search:{query.lower().strip()}:{limit}:{enrich}"
    cached = get_cached_result(cache_key)
    if cached is not None:
        return jsonify(cached)
    search_results = movies_df[movies_df['title'].str.contains(query, case=False, na=False)]
    if limit:
        search_results = search_results.head(limit)
    data = serialize_movies(search_results, enrich=enrich)
    set_cached_result(cache_key, data)
    return jsonify(data)

@app.route('/api/movies/top-rated', methods=['GET'])
def get_top_rated_movies():
    limit = parse_limit(request.args.get('limit'), default=10, max_value=50)
    enrich = request.args.get('enrich', default='posters')
    cache_key = f"top-rated:{limit}:{enrich}"
    cached = get_cached_result(cache_key)
    if cached is not None:
        return jsonify(cached)
    top_rated = movies_df.sort_values(by='weighted_rating', ascending=False).head(limit)
    data = serialize_movies(top_rated, enrich=enrich)
    set_cached_result(cache_key, data)
    return jsonify(data)

@app.route('/api/movies/movie-details', methods=['GET'])
def get_movie_details():
    movie_title = request.args.get('title')
    if not movie_title:
        return jsonify({'error': 'Title is required'}), 400
    movie = movies_df[movies_df['title'] == movie_title]
    return jsonify(serialize_movies(movie, enrich='full'))

@app.route('/api/movies/id/<int:movie_id>', methods=['GET'])
def get_movie_by_id(movie_id):
    movie = movies_df[movies_df['id'] == movie_id]
    if movie.empty:
        return jsonify({'error': 'Movie not found'}), 404
    return jsonify(serialize_movies(movie, enrich='full'))

if __name__ == '__main__':
    app.run(debug=True)