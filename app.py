import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds

# Set Page Configurations (Netflix-Themed Dark UI)
st.set_page_config(page_title="🎬 Movie Recommendation", page_icon="🎥", layout="wide")

# Apply Netflix-Style Dark Theme with Visible Text
st.markdown(
    """
    <style>
        .stApp {
            background-color: #141414;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: red;
        }
        .stButton>button {
            background-color: red;
            color: white;
            font-size: 16px;
            border-radius: 10px;
        }
        .stRadio label, .stSelectbox label, .stSlider label {
            font-size: 18px;
            color: white;
        }
        .stDataFrame {
            background-color: black;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to Load and Preprocess Data
@st.cache_data
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ Dataset not found at: {file_path}. Please check the file path.")
        return None

    df = pd.read_csv(file_path)

    # Convert Movie_Rating to numeric
    df["Movie_Rating"] = pd.to_numeric(df["Movie_Rating"], errors="coerce")

    # Encode Genre
    label_encoder = LabelEncoder()
    df["Genre_Encoded"] = label_encoder.fit_transform(df["Genre"])

    # Scale Year & Rating
    scaler = MinMaxScaler()
    df[["Year_Scaled", "Rating_Scaled"]] = scaler.fit_transform(df[["Year", "Movie_Rating"]])

    return df

# Function to Train KNN Model
@st.cache_resource
def train_knn(df):
    df["Rating_Scaled"] *= 2  # Increase rating influence

    features = df[["Year_Scaled", "Genre_Encoded", "Rating_Scaled"]]
    knn_model = NearestNeighbors(n_neighbors=10, metric="euclidean")
    knn_model.fit(features.to_numpy())
    
    return knn_model, features

# Function to Train SVD Model
@st.cache_resource
def train_svd(df):
    movie_features_matrix = df[["Year_Scaled", "Genre_Encoded", "Rating_Scaled"]].values
    k = min(movie_features_matrix.shape) - 1
    U, sigma, Vt = svds(movie_features_matrix, k=k)
    sigma = np.diag(sigma)
    predicted_matrix = np.dot(np.dot(U, sigma), Vt)
    return pd.DataFrame(predicted_matrix, index=df["Title"])

# Function to Determine Best Algorithm (KNN or SVD)
def select_best_algorithm():
    if len(df) > 500:  # Adjust based on dataset size
        return "SVD"
    return "KNN"

# Function for KNN Recommendations by Movie Features
def knn_recommend_by_title(movie_title, df, features, knn_model, n_recommendations=5):
    if movie_title.lower().strip() not in df["Title"].str.lower().str.strip().values:
        return None
    
    movie_index = df[df["Title"].str.lower().str.strip() == movie_title.lower().strip()].index[0]
    movie_features = np.array(features.iloc[movie_index]).reshape(1, -1)
    distances, indices = knn_model.kneighbors(movie_features, n_neighbors=n_recommendations + 1)
    
    return df.iloc[indices[0][1:]][["Title", "Year", "Genre", "Movie_Rating"]]

# Function for KNN Recommendations by Genre, Year, and Rating
def knn_recommend_by_features(genre, year, rating, df, features, knn_model, n_recommendations=10):
    if genre not in df["Genre"].values:
        return None

    genre_encoded = df[df["Genre"] == genre]["Genre_Encoded"].values[0]
    user_features = np.array([[year, genre_encoded, rating * 2]])  # Increased weight for rating
    
    distances, indices = knn_model.kneighbors(user_features, n_neighbors=n_recommendations * 2)
    recommendations = df.iloc[indices[0]][["Title", "Year", "Genre", "Movie_Rating"]]

    # 🔥 Strictly filter by selected genre, year, and allow rating ±0.5 range
    recommendations = recommendations[
        (recommendations["Genre"] == genre) &
        (recommendations["Year"] == year) &
        (recommendations["Movie_Rating"].between(rating - 0.5, rating + 0.5))
    ]

    # If still empty, remove year restriction and expand rating range
    if recommendations.empty:
        recommendations = df[
            (df["Genre"] == genre) &
            (df["Movie_Rating"].between(rating - 1, rating + 1))
        ]

    return recommendations.head(n_recommendations)


# Function for SVD Recommendations by Genre, Year, and Rating
def svd_recommend_by_features(genre, year, rating, df_svd, df, n_recommendations=10):
    if genre not in df["Genre"].values:
        return None

    genre_encoded = df[df["Genre"] == genre]["Genre_Encoded"].values[0]
    movie_vector = np.array([year, genre_encoded, rating * 2], dtype=float).reshape(1, -1)
    similarity_scores = df_svd.dot(movie_vector.T).squeeze()
    recommended_titles = similarity_scores.nlargest(n_recommendations * 2).index
    recommendations = df[df["Title"].isin(recommended_titles)][["Title", "Year", "Genre", "Movie_Rating"]]

    # 🔥 Strictly filter by selected genre, year, and allow rating ±0.5 range
    recommendations = recommendations[
        (recommendations["Genre"] == genre) &
        (recommendations["Year"] == year) &
        (recommendations["Movie_Rating"].between(rating - 0.5, rating + 0.5))
    ]

    # If still empty, remove year restriction and expand rating range
    if recommendations.empty:
        recommendations = df[
            (df["Genre"] == genre) &
            (df["Movie_Rating"].between(rating - 1, rating + 1))
        ]

    return recommendations.head(n_recommendations)


# Load Dataset with Correct File Path
file_path = "C:\\Users\\yukth\\Downloads\\Dataset_with_Movie_Ratings.csv"
df = load_and_preprocess_data(file_path)

if df is not None:
    knn_model, features = train_knn(df)
    df_svd = train_svd(df)

    # Auto-select best algorithm
    best_algo = select_best_algorithm()
    st.write(f"🔍 Using **{best_algo}** for recommendations.")

    # User Input for Recommendations
    method = st.radio("Select Recommendation Type:", ["By Movie Title", "By Genre, Year, and Rating"])

    if method == "By Movie Title":
        movie_title = st.text_input("Enter a Movie Title:")

        if st.button("Get Recommendations"):
            recommendations = knn_recommend_by_title(movie_title, df, features, knn_model)

            if recommendations is None or recommendations.empty:
                st.error("❌ No recommendations found. Try a different movie.")
            else:
                st.subheader(f"Movies Similar to {movie_title}:")
                st.dataframe(recommendations)

    elif method == "By Genre, Year, and Rating":
        genre = st.selectbox("Select Genre:", df["Genre"].unique())
        year = st.slider("Select Year:", int(df["Year"].min()), int(df["Year"].max()), step=1)
        rating = st.slider("Select Rating:", float(df["Movie_Rating"].min()), float(df["Movie_Rating"].max()), step=0.5)

        if st.button("Get Recommendations"):
            recommendations = knn_recommend_by_features(genre, year, rating, df, features, knn_model)

            if recommendations is None or recommendations.empty:
                st.error("❌ No recommendations found. Try adjusting the filters.")
            else:
                st.subheader(f"Recommended Movies Based on Selected Features:")
                st.dataframe(recommendations)
