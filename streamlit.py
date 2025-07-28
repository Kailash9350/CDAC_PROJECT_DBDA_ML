import streamlit as st
import requests
import nltk
import re
import joblib
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open("D:/DBDA/Machine_Learning_Project/final_model_lr.pkl", 'rb') as f:
    model = joblib.load(f)

with open("D:/DBDA/Machine_Learning_Project/tf_idf_vectorizer.pkl", 'rb') as f:
    vectorizer = joblib.load(f)

# TMDb API details
TMDB_API_KEY = '35250d55ae36d4d2136a5c928021acc1'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

# Function to clean review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Search movie
def search_movie(name):
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {'api_key': TMDB_API_KEY, 'query': name.strip()}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data['results'][0] if data.get('results') else None
    except:
        return None

# Get movie details
def get_movie_details(movie_id):
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {'api_key': TMDB_API_KEY, 'append_to_response': 'credits'}
    return requests.get(url, params=params).json()

# Get movie reviews
def get_reviews(movie_id):
    url = f"{TMDB_BASE_URL}/movie/{movie_id}/reviews"
    params = {'api_key': TMDB_API_KEY}
    response = requests.get(url, params=params).json()
    return [r['content'] for r in response.get('results', [])]

# Streamlit UI
st.set_page_config(page_title=" Movie Sentiment Analyzer", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'> Movie Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)

movie_name = st.text_input("Enter a Movie Name")

if movie_name:
    movie = search_movie(movie_name)

    if movie:
        movie_id = movie['id']
        details = get_movie_details(movie_id)
        reviews = get_reviews(movie_id)

        # Poster
        poster_path = details.get('poster_path')
        if poster_path:
            st.image(f"https://image.tmdb.org/t/p/w500{poster_path}", width=800)

        # Title and year
        title = details.get('title', 'Unknown Title')
        release_date = details.get('release_date', '')[:4]
        st.markdown(f"###  {title} ({release_date})")

        # Genres
        genres = [genre['name'] for genre in details.get('genres', [])]
        if genres:
            st.markdown(f" Genres: {', '.join(genres)}")

        # Overview
        st.markdown(" Overview")
        st.info(details.get('overview', 'No overview available.'))

        # Cast
        st.markdown(" Top Cast")
        cast = details.get('credits', {}).get('cast', [])
        if cast:
            top_cast = cast[:5]
            for member in top_cast:
                name = member.get('name')
                character = member.get('character')
                st.write(f"{name} as *{character}*")

        # Sentiment Analysis or fallback to rating
        st.markdown(" Sentiment Analysis")
        if reviews:
            cleaned_reviews = [clean_text(r) for r in reviews]
            X = vectorizer.transform(cleaned_reviews)
            preds = model.predict(X)
            pos_percent = (sum(preds == 1) / len(preds)) * 100
        else:
            vote_average = details.get('vote_average', 0)
            pos_percent = (vote_average / 10) * 100 - 5  # fallback with margin

        neg_percent = 100 - pos_percent

        st.success(f" Positive Sentiment: {pos_percent:.2f}%")
        st.error(f" Negative Sentiment: {neg_percent:.2f}%")

        # Final Recommendation
        st.markdown("---")
        if pos_percent >= 65:
            st.markdown("<h2 style='text-align: center; color: green;'> Recommended: Worth Watching!</h2>", unsafe_allow_html=True)
        elif pos_percent >= 50:
            st.markdown("<h2 style='text-align: center; color: orange;'> Mixed Reviews: Watch at Your Own Discretion</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: red;'> Not Recommended to watch </h2>", unsafe_allow_html=True)
    else:
        st.error("Movie not found. Please try another name.")
