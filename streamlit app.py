import streamlit as st
import pandas as pd
import numpy as np
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file, engine="openpyxl")

    df = load_data(uploaded_file)
else:
    st.warning("Please upload a valid Excel file with UserID, MovieTitle, Genre, Rating columns.")
    st.stop()



# Prepare user-item matrix
user_movie_ratings = df.pivot_table(index="UserID", columns="MovieTitle", values="Rating", aggfunc=np.mean)
user_movie_ratings_filled = user_movie_ratings.fillna(0)

@st.cache_data
def compute_svd(matrix):
    svd = TruncatedSVD(n_components=20, random_state=42)
    latent_matrix = svd.fit_transform(matrix)
    user_similarity = cosine_similarity(latent_matrix)
    return user_similarity

user_similarity = compute_svd(user_movie_ratings_filled)

def get_recommendations(user_id, top_n=5):
    user_index = user_id - 1  # zero-based index
    if user_index >= len(user_similarity) or user_index < 0:
        return []
    similarity_scores = user_similarity[user_index]
    similar_users = similarity_scores.argsort()[-top_n-1:-1][::-1]  # exclude self

    similar_user_ratings = user_movie_ratings.iloc[similar_users].mean(axis=0)
    unseen_movies = user_movie_ratings.iloc[user_index].isna()
    recommendations = similar_user_ratings[unseen_movies].sort_values(ascending=False).head(top_n)
    return recommendations.index.tolist()

# User input
user_ids = user_movie_ratings.index.tolist()
selected_user = st.selectbox("Select User ID:", user_ids)

if selected_user:
    recommendations = get_recommendations(selected_user)
    st.subheader(f"Top Movie Recommendations for User {selected_user}:")
    if recommendations:
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.write("No recommendations available for this user.")

    # Visualizations

    st.subheader("User Movie Ratings Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(user_movie_ratings_filled, cmap="coolwarm", cbar_kws={'label': 'Rating'}, ax=ax)
    ax.set_xlabel("Movie Title")
    ax.set_ylabel("User ID")
    st.pyplot(fig)

    st.subheader("Average Movie Rating by Genre")
    genre_rating = df.groupby("Genre")["Rating"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genre_rating.values, y=genre_rating.index, palette="viridis", ax=ax2)
    ax2.set_xlabel("Average Rating")
    ax2.set_ylabel("Genre")
    st.pyplot(fig2)

    if recommendations:
        st.subheader(f"Top {len(recommendations)} Recommended Movies Average Ratings")
        recommended_scores = [user_movie_ratings_filled[movie].mean() for movie in recommendations]
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=recommendations, y=recommended_scores, palette="magma", ax=ax3)
        ax3.set_xlabel("Movie")
        ax3.set_ylabel("Average Rating")
        ax3.set_xticklabels(recommendations, rotation=45)
        st.pyplot(fig3)
