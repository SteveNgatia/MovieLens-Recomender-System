import streamlit as st
import pandas as pd
import pickle as pkl
import random
import time
import joblib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import base64

movie_ratings_pivot_df = pd.read_pickle("movie_ratings_pivot_df")
sparse_matrix = csr_matrix(movie_ratings_pivot_df.values)

with open("model", "rb") as f:
    knn_model = pkl.load(f)



# Datasets
movies = pd.read_csv("movies.csv")

ratings = pd.read_csv("ratings.csv")
# Preprocessing: Convert list of genres to string
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
movies['genres_str'] = movies['genres'].apply(lambda x: ','.join(x))
# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres_str'])
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def recommend_movies(movie_title, cosine_sim=cosine_sim, df=movies, num_recommendations=5):
    """
    The function returns a list of recommended movies based on how similar the movies are to the one they have provided.
    """
    # Get the index of the movie with the given title
    idx = df[df['title'].str.contains(movie_title, case=False, regex=False)].index
    
    if len(idx) == 0:
        return "Movie not found in the database."

    idx = idx[0]

    # Get the pairwise similarity scores with other movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top similar movies indices (excluding the movie itself)
    sim_scores = sim_scores[1:num_recommendations+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top recommended movie titles as a list
    return list(df['title'].iloc[movie_indices])


def recommend_movies_for_user(user_id, num_recommendations=5):
    # Get the index of the user's column in the pivot table
    user_index = movie_ratings_pivot_df.columns.get_loc(user_id)

    # Get the movies already rated by the user
    watched_movies = movie_ratings_pivot_df.iloc[:, user_index]

    # Get the distances and indices of the nearest neighbors
    distances, indices = knn_model.kneighbors(sparse_matrix[user_index], n_neighbors=num_recommendations+1)

    # Exclude the user's own index (which is always the closest)
    indices = indices.squeeze()[1:]
    distances = distances.squeeze()[1:]

    # Filter out movies already watched by the user
    recommended_movie_indices = [index for index in indices if movie_ratings_pivot_df.iloc[index, user_index] == 0]

    # Get recommended movie titles
    recommended_movie_titles = movie_ratings_pivot_df.index[recommended_movie_indices].to_list()
    return recommended_movie_titles


def hybrid_recommendations(user_id, movie_title, num_recommendations=5):
    
    # Check if the user is new (has no interaction history)
    if user_id not in movie_ratings_pivot_df.columns:
        # Recommend popular movies instead
        popular_movies = movies['movieId'].value_counts().index[:num_recommendations].tolist()
        return movies[movies['movieId'].isin(popular_movies)]['title'].tolist()
    else:
        # Collaborative Filtering
        collaborative_recommendations = recommend_movies_for_user(user_id, num_recommendations=5)
        
        # Content-Based Filtering
        content_based_recommendations = recommend_movies(movie_title, cosine_sim=cosine_sim, df=movies, num_recommendations=5)
        
        # Combine recommendations from both methods
        hybrid_recommendations = set(collaborative_recommendations + content_based_recommendations)
        
        return list(hybrid_recommendations)

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
img = get_img_as_base64("img4.jpg")        
# Define Streamlit app content
def main():
    st.title("Movie Recommender App")
    page_bg_img = f"""
	<style>
	[data-testid="stAppViewContainer"] > .main {{
	background-image: url("data:image/png;base64,{img}");
	background-size: cover;
	background-position: top right;
	background-repeat: no-repeat;
	background-attachment: fixed;
    
	}}
    [data-testid="stWidgetLabel"] p {{
    color: black;
    font-size: 20px;    
    }}
    [data-testid="stMarkdownContainer"] p {{
    color: black;
    font-size: 20px;   
    }}
    [data-testid="stTickBarMax"] {{
    color: black;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
   [data-testid="stToolbar"] {{
   right: 2rem;
   }}


	</style>
	"""
    # Render status
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Movie Recommendations"])

    if page == "Home":
        st.write("Welcome to Movie Recommender App!")
        # st.write("Status: OK")

    elif page == "Movie Recommendations":
        st.subheader("Movie Recommendations")
        
        # Get user input
        movie_title = (st.text_input("Enter a movie title:", ""))
        user_id = int(st.text_input("Enter user ID:", 1000))
        
        if st.button("Get Recommendations"):
            if user_id:
                try:
                    # Get movie recommendations
                    recommendations = hybrid_recommendations(user_id,movie_title)
                    st.write("Recommended Movies:")
                    st.write(recommendations)
                except:
                    st.error("An error occurred. Please try again.")
            else:
                st.warning("Please enter both a movie title and a user ID.")

# Run the app
if __name__ == "__main__":
    main()

