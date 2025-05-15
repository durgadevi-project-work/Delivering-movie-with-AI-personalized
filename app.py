import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Personalized Movie Recommendation System")

# Upload the dataset
uploaded_file = st.file_uploader("Upload your movie rating dataset (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Prepare the user-item matrix
    user_movie_ratings = df.pivot_table(index="UserID", columns="MovieTitle", values="Rating", aggfunc=np.mean)
    user_movie_ratings_filled = user_movie_ratings.fillna(0)

    # Apply SVD
    svd = TruncatedSVD(n_components=20, random_state=42)
    latent_matrix = svd.fit_transform(user_movie_ratings_filled)
    user_similarity = cosine_similarity(latent_matrix)

    # Recommendation function
    def get_recommendations(user_id, top_n=5):
        user_index = user_id - 1  # zero-based index
        similarity_scores = user_similarity[user_index]
        similar_users = similarity_scores.argsort()[-top_n-1:-1][::-1]
        similar_user_ratings = user_movie_ratings.iloc[similar_users].mean(axis=0)
        unseen_movies = user_movie_ratings.iloc[user_index].isna()
        recommendations = similar_user_ratings[unseen_movies].sort_values(ascending=False).head(top_n)
        return recommendations.index.tolist()

    # Sidebar: user input
    st.sidebar.header("🔧 Customize")
    user_id = st.sidebar.slider("Select User ID", int(df["UserID"].min()), int(df["UserID"].max()), step=1)
    top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

    # Generate recommendations
    recommended_movies = get_recommendations(user_id, top_n)
    st.subheader(f"🎯 Top {top_n} Movie Recommendations for User {user_id}")
    for movie in recommended_movies:
        st.write(f"- {movie}")

    # Plots
    st.subheader("📊 Visualizations")

    # 1. Heatmap
    st.markdown("#### User-Movie Ratings Heatmap")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.heatmap(user_movie_ratings_filled, cmap="coolwarm", ax=ax1, cbar_kws={'label': 'Rating'})
    ax1.set_title("User Movie Ratings Heatmap")
    ax1.set_xlabel("Movie Title")
    ax1.set_ylabel("User ID")
    st.pyplot(fig1)

    # 2. Bar chart: Average rating per genre
    st.markdown("#### Average Rating by Genre")
    genre_rating = df.groupby("Genre")["Rating"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genre_rating.values, y=genre_rating.index, ax=ax2, palette="viridis")
    ax2.set_title("Average Movie Rating by Genre")
    ax2.set_xlabel("Average Rating")
    ax2.set_ylabel("Genre")
    st.pyplot(fig2)

    # 3. Bar chart: Top N recommended movies
    st.markdown(f"#### Top {top_n} Recommended Movies (Average Ratings)")
    recommended_scores = [user_movie_ratings_filled[movie].mean() for movie in recommended_movies]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=recommended_movies, y=recommended_scores, ax=ax3, palette="magma")
    ax3.set_title(f"Top {top_n} Recommended Movies for User {user_id}")
    ax3.set_xlabel("Movie")
    ax3.set_ylabel("Average Rating")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    # 4. Boxplot: Rating distribution by genre
    st.markdown("#### Rating Distribution by Genre")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="Genre", y="Rating", data=df, ax=ax4, palette="Set2")
    ax4.set_title("Rating Distribution by Genre")
    ax4.set_xlabel("Genre")
    ax4.set_ylabel("Rating")
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)
else:
    st.info("Please upload a dataset to get started.")
