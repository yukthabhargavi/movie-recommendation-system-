# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# %%
# Load and preprocess data
def load_and_preprocess_data(file_path):
    """Loads the dataset and applies preprocessing."""
    df = pd.read_csv(file_path)

    # Encode genres
    label_encoder = LabelEncoder()
    df["Genre_Encoded"] = label_encoder.fit_transform(df["Genre"])

    # Scale numerical features
    scaler = StandardScaler()
    df[["Year_Scaled", "Rating_Scaled"]] = scaler.fit_transform(df[["Year", "Movie_Rating"]])

    return df, label_encoder, scaler

# %%
# KNN recommendation function
def recommend_movies_knn(df, genre, year, rating, label_encoder, scaler):
    """
    Recommends movies based on genre, year, and rating using KNN.
    """

    # Step 1: Filter movies by the requested year
    df_filtered = df[df["Year"] == year].copy()
    if df_filtered.empty:
        return "No movies found for the specified year."

    # Step 2: Filter movies by genre (include multi-genre matches)
    df_filtered = df_filtered[df_filtered["Genre"].str.contains(genre, case=False, na=False)]
    if df_filtered.empty:
        return "No movies found for the specified genre in this year."

    # Step 3: Encode genre input using the same encoder from preprocessing
    genre_encoded = label_encoder.transform([genre])[0]  # Encode input genre

    # Step 4: Scale numerical features using the **existing** scaler
    scaler.fit(df[["Year", "Movie_Rating"]])  # Ensure the scaler is trained on original dataset
    df_filtered[["Year_Scaled", "Rating_Scaled"]] = scaler.transform(df_filtered[["Year", "Movie_Rating"]])

    # Step 5: Train KNN only on the filtered dataset (avoids index mismatch)
    knn_model = NearestNeighbors(n_neighbors=min(5, len(df_filtered)), metric="euclidean")
    features_filtered = df_filtered[["Year_Scaled", "Rating_Scaled"]]
    knn_model.fit(features_filtered.to_numpy())

    # Step 6: Prepare input feature vector (scale properly)
    input_features = np.array([[year, rating]])  # Ensure input is a 2D array
    input_features_scaled = scaler.transform(input_features)  # Apply proper scaling

    # Step 7: Use KNN to find similar movies
    distances, indices = knn_model.kneighbors(input_features_scaled)

    # Retrieve recommended movies based on indices
    recommended_movies = df_filtered.iloc[indices[0]][["Title", "Year", "Genre", "Movie_Rating"]]

    return recommended_movies

# %%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

def recommend_movies_svd(df, movie_title, n_recommendations=5):
    """
    Finds movies similar to the given movie title using Singular Value Decomposition (SVD).
    Returns Title, Year, Genre, and Movie Rating.
    """

    # Create a movie-rating matrix (Movies × Genres)
    rating_matrix = df.pivot_table(index="Title", columns="Genre", values="Movie_Rating", aggfunc="mean").fillna(0)

    if movie_title not in rating_matrix.index:
        return f"Movie title '{movie_title}' not found in dataset."

    # Convert to numpy array for SVD
    matrix = rating_matrix.to_numpy(dtype=np.float64)  # Ensure matrix is float

    # Ensure valid k-value
    num_movies, num_features = matrix.shape
    if num_movies < 2 or num_features < 2:
        return "Not enough data to perform SVD-based recommendations."

    k_value = min(10, num_movies - 1, num_features - 1)  # Adjust k dynamically

    # Perform Singular Value Decomposition
    U, sigma, Vt = svds(matrix, k=k_value)  
    sigma = np.diag(sigma)

    # Compute item similarity using cosine similarity
    movie_factors = np.dot(U, sigma)  
    similarity_matrix = cosine_similarity(movie_factors)

    # Convert to DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=rating_matrix.index, columns=rating_matrix.index)

    # Get top N most similar movies (excluding itself)
    similar_movies = similarity_df[movie_title].sort_values(ascending=False).iloc[1:n_recommendations + 1]

    # Retrieve movie details (Title, Year, Genre, Rating)
    recommended_movies = df[df["Title"].isin(similar_movies.index)][["Title", "Year", "Genre", "Movie_Rating"]]

    return recommended_movies


# %%
# Load the dataset
file_path = r"C:\Users\yukth\Downloads\Dataset_with_Movie_Ratings.csv"
df, label_encoder, scaler = load_and_preprocess_data(file_path)

# Ask the user for recommendations based on title or features
recommendation_type = input("Do you want recommendations based on a movie title or movie features? (Enter 'title' or 'features'): ").strip().lower()

if recommendation_type == "title":
    # Get user input for SVD recommendations
    movie_title_input = input("Enter a movie title for SVD-based recommendations: ").strip()

    # Get SVD recommendations
    svd_recommendations = recommend_movies_svd(df, movie_title_input)

    # Display results
    print("\nRecommended Movies (SVD - Based on Movie Title):")
    print(svd_recommendations)

elif recommendation_type == "features":
    # Get user input for KNN recommendations
    genre_input = input("Enter the genre: ").strip()
    year_input = int(input("Enter the year: "))
    rating_input = float(input("Enter the rating (1-5): "))

    # Get KNN recommendations
    knn_recommendations = recommend_movies_knn(df, genre=genre_input, year=year_input, rating=rating_input, 
                                               label_encoder=label_encoder, scaler=scaler)

    # Display results
    print("\nRecommended Movies (KNN - Based on Features):")
    print(knn_recommendations)

else:
    print("Invalid input. Please enter 'title' or 'features'.")

# %%



