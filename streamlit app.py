import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.title("Personalized Movie Recommendation System")

uploaded_file = st.file_uploader("Upload your movie rating Excel file", type=["xlsx"])
if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file, engine='openpyxl')

    # Prepare user-item matrix
    user_movie_ratings = df.pivot_table(index="UserID", columns="MovieTitle", values="Rating", aggfunc=np.mean)
    user_movie_ratings_filled = user_movie_ratings.fillna(0)

    # Compute SVD and similarity only once, cache it
    @st.cache_data
    def compute_similarity(ratings_matrix):
        svd = TruncatedSVD(n_components=20, random_state=42)
        latent_matrix = svd.fit_transform(ratings_matrix)
        return cosine_similarity(latent_matrix)

    user_similarity = compute_similarity(user_movie_ratings_filled)

    def get_recommendations(user_id, top_n=5):
        user_index = user_id - 1
        if user_index < 0 or user_index >= len(user_similarity):
            return []
        similarity_scores = user_similarity[user_index]
        similar_users = similarity_scores.argsort()[-top_n-1:-1][::-1]  # exclude self
        similar_user_ratings = user_movie_ratings.iloc[similar_users].mean(axis=0)
        unseen_movies = user_movie_ratings.iloc[user_index].isna()
        recommendations = similar_user_ratings[unseen_movies].sort_values(ascending=False).head(top_n)
        return recommendations.index.tolist()

    # Select user from available UserIDs
    user_ids = user_movie_ratings.index.tolist()
    selected_user = st.selectbox("Select User ID", user_ids)

    if selected_user:
        recommended_movies = get_recommendations(selected_user)
        st.subheader(f"Top {len(recommended_movies)} Movie Recommendations for User {selected_user}:")
        if recommended_movies:
            for movie in recommended_movies:
                st.write(f"- {movie}")
        else:
            st.write("No recommendations available for this user.")

        # Heatmap of user ratings
        st.subheader("User Movie Ratings Heatmap")
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        sns.heatmap(user_movie_ratings_filled, cmap="coolwarm", cbar_kws={'label': 'Rating'}, ax=ax1)
        ax1.set_xlabel("Movie Title")
        ax1.set_ylabel("User ID")
        st.pyplot(fig1)

        # Average rating per genre bar chart
        if "Genre" in df.columns:
            st.subheader("Average Movie Rating by Genre")
            genre_rating = df.groupby("Genre")["Rating"].mean().sort_values()
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=genre_rating.values, y=genre_rating.index, palette="viridis", ax=ax2)
            ax2.set_xlabel("Average Rating")
            ax2.set_ylabel("Genre")
            st.pyplot(fig2)

        # Recommended movies average rating bar chart
        if recommended_movies:
            st.subheader("Recommended Movies Average Ratings")
            recommended_scores = [user_movie_ratings_filled[movie].mean() for movie in recommended_movies]
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=recommended_movies, y=recommended_scores, palette="magma", ax=ax3)
            ax3.set_xlabel("Movie")
            ax3.set_ylabel("Average Rating")
            ax3.set_xticklabels(recommended_movies, rotation=45)
            st.pyplot(fig3)

        # Boxplot of ratings by genre
        if "Genre" in df.columns:
            st.subheader("Rating Distribution by Genre")
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            sns.boxplot(x="Genre", y="Rating", data=df, palette="Set2", ax=ax4)
            ax4.set_xlabel("Genre")
            ax4.set_ylabel("Rating")
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
            st.pyplot(fig4)

else:
    st.info("Please upload an Excel file containing UserID, MovieTitle, Genre, and Rating columns.")
