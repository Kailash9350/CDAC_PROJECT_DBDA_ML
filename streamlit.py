import streamlit as st
import requests
import nltk
import re
import joblib
from nltk.corpus import stopwords 
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

with open("final_model_lr.pkl", 'rb') as f:
    model=joblib.load(f)
with open("tf_idf_vectorizer.pkl", 'rb') as f:
    vectorizer=joblib.load(f)

TMDB_API_KEY='35250d55ae36d4d2136a5c928021acc1'
TMDB_BASE_URL='https://api.themoviedb.org/3'

def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+", "", text)
    text=re.sub(r"[^a-z\s]", "", text)
    tokens=text.split()
    tokens=[word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def search_movie(name):
    url=f"{TMDB_BASE_URL}/search/movie"
    params={'api_key': TMDB_API_KEY, 'query': name.strip()}
    try:
        response=requests.get(url, params=params)
        response.raise_for_status()
        data=response.json()
        return data['results'][0] if data.get('results') else None
    except:
        return None

def get_movie_details(movie_id):
    url=f"{TMDB_BASE_URL}/movie/{movie_id}"
    params={'api_key': TMDB_API_KEY, 'append_to_response': 'credits'}
    return requests.get(url, params=params).json()

def get_reviews(movie_id):
    url=f"{TMDB_BASE_URL}/movie/{movie_id}/reviews"
    params={'api_key': TMDB_API_KEY}
    response=requests.get(url, params=params).json()
    return [r['content'] for r in response.get('results', [])] 

def sentiment_chart(positive_percent, negative_percent, title):
    fig, ax=plt.subplots()
    ax.bar(['Positive', 'Negative'], [positive_percent, negative_percent], color=['green', 'red'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title(title)
    for i, val in enumerate([positive_percent, negative_percent]):
        ax.text(i, val + 1, f"{val:.2f}%", ha='center', fontsize=10)
    st.pyplot(fig)

st.set_page_config(page_title="Movie Review Sentiment Analysis", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'> Movie Review Sentiment Analysis</h1>",
    unsafe_allow_html=True)
movie_name=st.text_input("Enter a Movie Name")
if movie_name:
    movie=search_movie(movie_name)
    if movie:
        movie_id=movie['id']
        details=get_movie_details(movie_id)
        reviews=get_reviews(movie_id)

        poster_path=details.get('poster_path')
        if poster_path:
            st.image(f"https://image.tmdb.org/t/p/w500{poster_path}", width=800)

        title=details.get('title', 'Unknown Title')
        release_date=details.get('release_date', '')[:4]
        st.markdown(f"###  {title} ({release_date})")

        genres=[genre['name'] for genre in details.get('genres', [])]
        if genres:
            st.markdown(f" Genres: {', '.join(genres)}")
        st.markdown("Overview")
        st.info(details.get('overview', 'No overview available'))

        st.markdown("Top Cast")
        cast=details.get('credits', {}).get('cast', [])
        if cast:
            top_cast=cast[:5]
            for member in top_cast:
                name=member.get('name')
                character=member.get('character')
                st.write(f"{name} as *{character}*")

        st.markdown(" Sentiment Analysis")
        if reviews:
            cleaned_reviews=[clean_text(r) for r in reviews]
            X=vectorizer.transform(cleaned_reviews)
            preds=model.predict(X)
            positive_percent=(sum(preds == 1) / len(preds)) * 100
        else:
            vote_average=details.get('vote_average', 0)
            positive_percent=(vote_average / 10) * 100 - 5

        negative_percent=100 - positive_percent

        st.success(f" Positive Sentiment: {positive_percent:.2f}%")
        st.error(f" Negative Sentiment: {negative_percent:.2f}%")
        sentiment_chart(positive_percent, negative_percent, "Movie Sentiment")

    
        st.markdown("---")
        if positive_percent >= 65:
            st.markdown("<h2 style='text-align: center; color: green;'> Recommended: Worth Watching</h2>", unsafe_allow_html=True)
        elif positive_percent >= 50:
            st.markdown("<h2 style='text-align: center; color: orange;'> Mixed Reviews</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: red;'> Not Recommended to watch </h2>", unsafe_allow_html=True)

        st.markdown("---")
        user_review=st.text_area("Write your review here:")
        if user_review.strip():
            cleaned_review=clean_text(user_review)
            X_user=vectorizer.transform([cleaned_review])
            pred_user=model.predict(X_user)[0]
            prob=model.predict_proba(X_user)[0][1] * 100
            positive_custom=prob
            negative_custom=100 - prob

            if pred_user==1:
                st.success(f"Your review is predicted as **Positive** ({prob:.2f}% confidence).")
            else:
                st.error(f"Your review is predicted as **Negative** ({100 - prob:.2f}% confidence).")

            sentiment_chart(positive_custom, negative_custom, "Your Review Sentiment")
    else:
        st.error("Movie not found")
