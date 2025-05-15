import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="AI Movie Recommender", layout="wide")
st.title("🎬 AI-Powered Personalized Movie Recommendation")

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_excel("personalized_movie_recommendation_dataset.xlsx", engine="openpyxl")
    return df

# Load dataset
df = load_data()

# Prepare the user-item matrix
user_movie_ratings = df.pivot_table(index="UserID", columns="MovieTitle", values="Rating", aggfunc=np.mean)
user_movie_ratings_filled = user_movie_ratings.fillna(0)

# Apply SVD and calculate user similarity
@st.cache_data
def compute_user_similarity(matrix):
    svd = TruncatedSVD(n_components=20, random_state=42)
    latent_matrix = svd.fit_transform(matrix)
    return cosine_similarity(latent_matrix)

user_similarity = compute_user_similarity(user_movie_ratings_filled)

# Recommendation function
def get_recommendations(user_id, top_n=5):
    user_index = user_id - 1
    similarity_scores = user_similarity[user_index]
    similar_users = similarity_scores.argsort()[-top_n-1:-1][::-1]

    similar_user_ratings = user_movie_ratings.iloc[similar_users].mean(axis=0)
    unseen_movies = user_movie_ratings.iloc[user_index].isna()
    recommendations = similar_user_ratings[unseen_movies].sort_values(ascending=False).head(top_n)
    return recommendations

# Sidebar for user selection
user_ids = user_movie_ratings.index.tolist()
selected_user = st.sidebar.selectbox("Select a User ID", user_ids)

# Display recommendations
recommendations = get_recommendations(selected_user)
st.subheader(f"🎯 Top Movie Recommendations for User {selected_user}")
if not recommendations.empty:
    st.write(recommendations.to_frame(name="Predicted Rating"))
else:
    st.write("No recommendations available for this user.")

# --- Visualizations ---

# 1. Heatmap: User's movie ratings
st.subheader("🔥 Heatmap of User-Movie Ratings")
fig1, ax1 = plt.subplots(figsize=(14, 8))
sns.heatmap(user_movie_ratings_filled, cmap="coolwarm", cbar_kws={'label': 'Rating'}, ax=ax1)
ax1.set_xlabel("Movie Title")
ax1.set_ylabel("User ID")
st.pyplot(fig1)

# 2. Bar Graph: Average rating per genre
st.subheader("📊 Average Rating by Genre")
genre_rating = df.groupby("Genre")["Rating"].mean().sort_values()
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x=genre_rating.values, y=genre_rating.index, palette="viridis", ax=ax2)
ax2.set_xlabel("Average Rating")
ax2.set_ylabel("Genre")
st.pyplot(fig2)

# 3. Bar Graph: Top recommended movies
if not recommendations.empty:
    st.subheader("🌟 Top Recommended Movies (Average Ratings)")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=recommendations.index, y=[user_movie_ratings_filled[movie].mean() for movie in recommendations.index],
                palette="magma", ax=ax3)
    ax3.set_ylabel("Average Rating")
    ax3.set_xlabel("Movie Title")
    ax3.set_xticklabels(recommendations.index, rotation=45)
    st.pyplot(fig3)
