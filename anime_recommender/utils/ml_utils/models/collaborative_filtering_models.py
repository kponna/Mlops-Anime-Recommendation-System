import sys  
import pandas as pd
from anime_recommender.loggers.logging import logging
from anime_recommender.exception.exception import AnimeRecommendorException
 
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class CollaborativeAnimeRecommender:
    def __init__(self, df):
        self.df = df
        self.svd = None
        self.knn_item_based = None
        self.knn_user_based = None
        self.prepare_data()
 

    def prepare_data(self): 
        self.df = self.df.drop_duplicates()
        reader = Reader(rating_scale=(1, 10))
        self.data = Dataset.load_from_df(self.df[['user_id', 'anime_id', 'rating']], reader)
        self.anime_pivot = self.df.pivot_table(index='name', columns='user_id', values='rating').fillna(0)
        self.user_pivot = self.df.pivot_table(index='user_id', columns='name', values='rating').fillna(0)

    def train_svd(self):
        self.svd = SVD()
        cross_validate(self.svd, self.data, cv=5)
        trainset = self.data.build_full_trainset()
        self.svd.fit(trainset)  

    def train_knn_item_based(self):
        item_user_matrix = csr_matrix(self.anime_pivot.values)
        self.knn_item_based = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn_item_based.fit(item_user_matrix) 

    def train_knn_user_based(self):
        """Train the KNN model for user-based recommendations."""
        user_item_matrix = csr_matrix(self.user_pivot.values)
        self.knn_user_based = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn_user_based.fit(user_item_matrix)  
 
    def print_unique_user_ids(self):
        """Print unique user IDs from the dataset."""
        unique_user_ids = self.df['user_id'].unique()
        logging.info(f"Unique User IDs: {unique_user_ids}")
        return unique_user_ids
    
    def get_svd_recommendations(self, user_id, n=10, svd_model=None): 
        # Use the provided SVD model or the trained self.svd model
        svd_model = svd_model or self.svd
        if svd_model is None:
            raise ValueError("SVD model is not provided or trained.")

        # Ensure user exists in the dataset
        if user_id not in self.df['user_id'].unique():
            return f"User ID '{user_id}' not found in the dataset."

        # Get unique anime IDs
        anime_ids = self.df['anime_id'].unique()

        # Predict ratings for all anime for the given user
        predictions = [(anime_id, svd_model.predict(user_id, anime_id).est) for anime_id in anime_ids]
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Extract top N anime IDs
        recommended_anime_ids = [pred[0] for pred in predictions[:n]]

        # Get details of recommended anime
        recommended_anime = self.df[self.df['anime_id'].isin(recommended_anime_ids)].drop_duplicates(subset='anime_id')
        logging.info(f"Shape of recommended_anime: {recommended_anime.shape}")
        # Limit to N recommendations
        recommended_anime = recommended_anime.head(n)

        return pd.DataFrame({
            'Anime Name': recommended_anime['name'].values,
            'Genres': recommended_anime['genres'].values,
            'Image URL': recommended_anime['image url'].values,
            'Rating': recommended_anime['average_rating'].values
        })
 
    def get_item_based_recommendations(self, anime_name, n_recommendations=10, knn_item_model=None):
        # Use the provided model or fall back to self.knn_item_based
        knn_item_based = knn_item_model or self.knn_item_based
        if knn_item_based is None:
            raise ValueError("Item-based KNN model is not provided or trained.")

        # Ensure the anime name exists in the pivot table
        if anime_name not in self.anime_pivot.index:
            return f"Anime title '{anime_name}' not found in the dataset."

        # Get the index of the anime in the pivot table
        query_index = self.anime_pivot.index.get_loc(anime_name)

        # Use the KNN model to find similar animes (n_neighbors + 1 to exclude the query itself)
        distances, indices = knn_item_based.kneighbors(
            self.anime_pivot.iloc[query_index, :].values.reshape(1, -1),
            n_neighbors=n_recommendations + 1  # +1 because the query anime itself is included
        ) 
        recommendations = []
        for i in range(1, len(distances.flatten())):  # Start from 1 to exclude the query anime
            anime_title = self.anime_pivot.index[indices.flatten()[i]]
            distance = distances.flatten()[i]
            recommendations.append((anime_title, distance))

        # Fetch the recommended anime names (top n_recommendations)
        recommended_anime_titles = [rec[0] for rec in recommendations]
        logging.info(f"Top {n_recommendations} recommendations: {recommended_anime_titles}")
        filtered_df = self.df[self.df['name'].isin(recommended_anime_titles)].drop_duplicates(subset='name')
        logging.info(f"Shape of filtered df: {filtered_df.shape}")
        # Limit the results to `n_recommendations`
        filtered_df = filtered_df.head(n_recommendations)

        return pd.DataFrame({
            'Anime Name': filtered_df['name'].values,
            'Image URL': filtered_df['image url'].values,
            'Genres': filtered_df['genres'].values,
            'Rating': filtered_df['average_rating'].values
        }) 

    def get_user_based_recommendations(self, user_id, n_recommendations=10, knn_user_model=None):
        """
        Recommend anime for a given user based on similar users' preferences using the provided or trained KNN model.

        Args:
            user_id (int): The ID of the user.
            n_recommendations (int): Number of recommendations to return.
            knn_user_model (NearestNeighbors, optional): Pre-trained KNN model. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing recommended anime titles and related information.
        """
        # Use the provided model or fall back to self.knn_user_based
        knn_user_based = knn_user_model or self.knn_user_based
        if knn_user_based is None:
            raise ValueError("User-based KNN model is not provided or trained.")

        # Ensure the user exists in the pivot table
        user_id = float(user_id)  # Convert to match pivot table index type
        if user_id not in self.user_pivot.index:
            return f"User ID '{user_id}' not found in the dataset."

        # Find the user's index in the pivot table
        user_idx = self.user_pivot.index.get_loc(user_id)

        # Use the KNN model to find the nearest neighbors
        distances, indices = knn_user_based.kneighbors(
            self.user_pivot.iloc[user_idx, :].values.reshape(1, -1),
            n_neighbors=n_recommendations + 1  # Include the user itself
        )

        # Get the list of anime the user has already rated
        user_rated_anime = set(self.user_pivot.columns[self.user_pivot.iloc[user_idx, :] > 0])

        # Collect all anime rated by the nearest neighbors
        all_neighbor_ratings = []
        for i in range(1, len(distances.flatten())):  # Start from 1 to exclude the user itself
            neighbor_idx = indices.flatten()[i]
            neighbor_rated_anime = self.user_pivot.iloc[neighbor_idx, :]
            neighbor_ratings = neighbor_rated_anime[neighbor_rated_anime > 0]
            all_neighbor_ratings.extend(neighbor_ratings.index)

        # Count how frequently each anime is rated by neighbors
        anime_counter = Counter(all_neighbor_ratings)

        # Recommend anime not already rated by the user
        recommendations = [(anime, count) for anime, count in anime_counter.items() if anime not in user_rated_anime]
        recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by frequency

        # Extract recommended anime names and their details
        recommended_anime_titles = [rec[0] for rec in recommendations[:n_recommendations]]
        filtered_df = self.df[self.df['name'].isin(recommended_anime_titles)].drop_duplicates(subset='name')
        logging.info(f"Shape of filtered df: {filtered_df.shape}")
        # Limit the results to `n_recommendations`
        filtered_df = filtered_df.head(n_recommendations)

        return pd.DataFrame({
            'Anime Name': filtered_df['name'].values,
            'Image URL': filtered_df['image url'].values,
            'Genres': filtered_df['genres'].values,
            'Rating': filtered_df['average_rating'].values
        })