import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit Page Configuration
st.set_page_config(page_title="🎬 Movie Recommendation", page_icon="🎥", layout="wide")

# Apply Netflix-Style Dark Theme
st.markdown(
    """
    <style>
        .stApp {
            background-color: #141414;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: red;
            text-align: center;
        }
        .stButton>button {
            background-color: red;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            width: 100%;
            padding: 10px;
        }
        .stRadio label, .stSelectbox label, .stSlider label {
            font-size: 18px;
            color: white;
        }
        .stDataFrame {
            background-color: black;
            color: white;
        }
        .container {
            text-align: center;
            padding: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown("<h1 style='text-align: center; font-size: 36px;'>🎬 Movie Recommendation System 🎥</h1>", unsafe_allow_html=True)

# Load and preprocess data
file_url = "https://github.com/yukthabhargavi/movie-recommendation-system-/blob/main/Dataset_with_Movie_Ratings%20(1).csv"
df = pd.read_csv(file_url)

# Encode genres
label_encoder = LabelEncoder()
df["Genre_Encoded"] = label_encoder.fit_transform(df["Genre"])

# Scale numerical features
scaler = StandardScaler()
df[["Year_Scaled", "Rating_Scaled"]] = scaler.fit_transform(df[["Year", "Movie_Rating"]])

# KNN-based recommendations
def recommend_movies_knn(df, genre, year, rating, label_encoder, scaler):
    df_filtered = df[(df["Year"] == year) & (df["Movie_Rating"].between(rating - 0.25, rating + 0.25))].copy()
    if df_filtered.empty:
        return None

    df_filtered = df_filtered[df_filtered["Genre"].str.contains(genre, case=False, na=False)]
    if df_filtered.empty:
        return None

    genre_encoded = label_encoder.transform([genre])[0]
    df_filtered[["Year_Scaled", "Rating_Scaled"]] = scaler.transform(df_filtered[["Year", "Movie_Rating"]])

    knn_model = NearestNeighbors(n_neighbors=min(5, len(df_filtered)), metric="euclidean")
    features_filtered = df_filtered[["Year_Scaled", "Rating_Scaled"]]
    knn_model.fit(features_filtered.to_numpy())

    input_features_scaled = scaler.transform(pd.DataFrame([[year, rating]], columns=["Year", "Movie_Rating"]))
    distances, indices = knn_model.kneighbors(input_features_scaled)
    
    return df_filtered.iloc[indices[0]][["Title", "Year", "Genre", "Movie_Rating"]]

# SVD-based recommendations
def recommend_movies_svd(df, movie_title, n_recommendations=5):
    rating_matrix = df.pivot_table(index="Title", columns="Genre", values="Movie_Rating", aggfunc="mean").fillna(0)
    if movie_title not in rating_matrix.index:
        return None

    matrix = rating_matrix.to_numpy(dtype=np.float64)
    k_value = min(10, matrix.shape[0] - 1, matrix.shape[1] - 1)
    U, sigma, Vt = svds(matrix, k=k_value)
    similarity_matrix = cosine_similarity(np.dot(U, np.diag(sigma)))
    similarity_df = pd.DataFrame(similarity_matrix, index=rating_matrix.index, columns=rating_matrix.index)
    similar_movies = similarity_df[movie_title].sort_values(ascending=False).iloc[1:n_recommendations + 1]
    return df[df["Title"].isin(similar_movies.index)][["Title", "Year", "Genre", "Movie_Rating"]]

method = st.radio("Select Recommendation Type:", ["By Movie Title", "By Genre, Year, and Rating"], horizontal=True)

if method == "By Movie Title":
    movie_title = st.text_input("Enter a Movie Title:")
    if st.button("Get Recommendations"):
        recommendations = recommend_movies_svd(df, movie_title)
        if recommendations is None or recommendations.empty:
            st.error("❌ No recommendations found. Try a different movie.")
        else:
            st.subheader(f"Movies Similar to {movie_title}:")
            st.dataframe(recommendations)

elif method == "By Genre, Year, and Rating":
    genre = st.selectbox("Select Genre:", df["Genre"].unique())
    year = st.slider("Select Year:", int(df["Year"].min()), int(df["Year"].max()), step=1)
    rating = st.slider("Select Rating:", float(df["Movie_Rating"].min()), float(df["Movie_Rating"].max()), step=0.1)
    if st.button("Get Recommendations"):
        recommendations = recommend_movies_knn(df, genre, year, rating, label_encoder, scaler)
        if recommendations is None or recommendations.empty:
            st.error("❌ No recommendations found. Try adjusting the filters.")
        else:
            st.subheader("Recommended Movies:")
            st.dataframe(recommendations)
