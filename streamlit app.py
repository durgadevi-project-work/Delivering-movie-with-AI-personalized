import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎥 Personalized Movie Recommendation System")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Movie Ratings Excel File", type=["xlsx"])

if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Show raw data if user wants
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("📄 Raw Dataset")
        st.dataframe(df)

    # Pivot to user-movie rating matrix
    user_movie_ratings = df.pivot_table(index="UserID", columns="MovieTitle", values="Rating_
