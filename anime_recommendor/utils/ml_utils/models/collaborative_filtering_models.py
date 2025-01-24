import sys  
import pandas as pd
from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException
 
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
        user_item_matrix = csr_matrix(self.user_pivot.values)
        self.knn_user_based = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn_user_based.fit(user_item_matrix) 

    def get_svd_recommendations(self, user_id, n=10, svd_model=None):
        # Use the provided model or fall back to self.svd
        svd = svd_model or self.svd
        if svd is None:
            raise ValueError("SVD model is not provided or trained.")

        # Generate predictions for all anime IDs
        anime_ids = self.df['anime_id'].unique()
        predictions = [(anime_id, svd.predict(user_id, anime_id).est) for anime_id in anime_ids]
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Extract top anime IDs
        top_anime_ids = [pred[0] for pred in predictions[:n]]

        # Validate IDs against the dataset
        if not all(anime_id in self.df['anime_id'].values for anime_id in top_anime_ids):
            return "Some anime IDs are invalid or not present in the dataset."

        anime_indices = self.df[self.df['anime_id'].isin(top_anime_ids)].index
        return pd.DataFrame({
            'Anime name': self.df['name'].iloc[anime_indices].values,
            'Image URL': self.df['image url'].iloc[anime_indices].values,
            'Genres': self.df['genres'].iloc[anime_indices].values,
            'Rating': self.df['average_rating'].iloc[anime_indices].values
        })


    def get_item_based_recommendations(self, anime_name, n_recommendations=5, knn_item_model=None):
        # Use the provided model or fall back to self.knn_item_based
        knn_item_based = knn_item_model or self.knn_item_based
        if knn_item_based is None:
            raise ValueError("Item-based KNN model is not provided or trained.")

        # Ensure the anime name exists in the pivot table
        if anime_name not in self.anime_pivot.index:
            return f"Anime title '{anime_name}' not found in the dataset."

        # Get the index of the anime in the pivot table
        query_index = self.anime_pivot.index.get_loc(anime_name)

        # Use the KNN model to find similar animes
        distances, indices = knn_item_based.kneighbors(
            self.anime_pivot.iloc[query_index, :].values.reshape(1, -1),
            n_neighbors=n_recommendations + 1  # +1 because the query anime itself is included
        )

        # Map the indices to the original DataFrame
        try:
            anime_indices = self.df.loc[
                self.df['name'].isin(self.anime_pivot.index[indices.flatten()[1:]])
            ].index
        except Exception as e:
            return f"Error in fetching recommendations: {e}"

        # Return recommendations as a DataFrame
        return pd.DataFrame({
            'Anime Name': self.df.loc[anime_indices, 'name'].values,
            'Image URL': self.df.loc[anime_indices, 'image url'].values,
            'Genres': self.df.loc[anime_indices, 'genres'].values,
            'Rating': self.df.loc[anime_indices, 'average_rating'].values
        })


    def get_user_based_recommendations(self, user_id, n_recommendations=5, knn_user_model=None):
        # Use the provided model or fall back to self.knn_user_based
        knn_user_based = knn_user_model or self.knn_user_based
        if knn_user_based is None:
            raise ValueError("User-based KNN model is not provided or trained.")

        user_id = float(user_id)
        if user_id not in self.user_pivot.index:
            return f"User '{user_id}' not found in the dataset."

        user_idx = self.user_pivot.index.get_loc(user_id)
        distances, indices = knn_user_based.kneighbors(
            self.user_pivot.iloc[user_idx, :].values.reshape(1, -1),
            n_neighbors=n_recommendations + 1
        )

        user_rated_anime = set(self.user_pivot.columns[self.user_pivot.iloc[user_idx, :] > 0])
        all_neighbor_ratings = []
        for i in range(1, len(distances.flatten())):
            neighbor_idx = indices.flatten()[i]
            neighbor_rated_anime = self.user_pivot.iloc[neighbor_idx, :]
            neighbor_ratings = neighbor_rated_anime[neighbor_rated_anime > 0]
            all_neighbor_ratings.extend(neighbor_ratings.index)

        anime_counter = Counter(all_neighbor_ratings)
        recommendations = [(anime, count) for anime, count in anime_counter.items() if anime not in user_rated_anime]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_anime_names = [rec[0] for rec in recommendations[:n_recommendations]]
        anime_indices = self.df[self.df['name'].isin(top_anime_names)].index

        return pd.DataFrame({
            'Anime name': self.df['name'].iloc[anime_indices].values,
            'Image URL': self.df['image url'].iloc[anime_indices].values,
            'Genres': self.df['genres'].iloc[anime_indices].values,
            'Rating': self.df['rating'].iloc[anime_indices].values
        })



    # def get_svd_recommendations(self, user_id, n=10):
    #     if not self.svd:
    #         self.train_svd()
        
    #     # Generate predictions for all anime IDs
    #     anime_ids = self.df['anime_id'].unique() 
    #     predictions = [(anime_id, self.svd.predict(user_id, anime_id).est) for anime_id in anime_ids]
    #     predictions.sort(key=lambda x: x[1], reverse=True) 
    
    #     # Extract top anime IDs
    #     top_anime_ids = [pred[0] for pred in predictions[:n]]
        
    #     # Validate IDs against the dataset
    #     if not all(anime_id in self.df['anime_id'].values for anime_id in top_anime_ids):
    #         return "Some anime IDs are invalid or not present in the dataset."
 
    #     anime_indices = self.df[self.df['anime_id'].isin(top_anime_ids)].index
    #     return pd.DataFrame({
    #         'Anime name': self.df['name'].iloc[anime_indices].values,
    #         'Image URL': self.df['image url'].iloc[anime_indices].values,
    #         'Genres': self.df['genres'].iloc[anime_indices].values,
    #         'Rating': self.df['average_rating'].iloc[anime_indices].values
    #     }) 
    
    

    # def get_item_based_recommendations(self, anime_name, n_recommendations=5):
    #     # Check if the KNN model is trained; if not, train it
    #     if not self.knn_item_based:
    #         self.train_knn_item_based()
        
    #     # Ensure the anime name exists in the pivot table
    #     if anime_name not in self.anime_pivot.index:
    #         return f"Anime title '{anime_name}' not found in the dataset."
        
    #     # Get the index of the anime in the pivot table
    #     query_index = self.anime_pivot.index.get_loc(anime_name)
        
    #     # Use the KNN model to find similar animes
    #     distances, indices = self.knn_item_based.kneighbors(
    #         self.anime_pivot.iloc[query_index, :].values.reshape(1, -1), 
    #         n_neighbors=n_recommendations + 1  # +1 because the query anime itself is included
    #     )
        
    #     # Map the indices to the original DataFrame
    #     try:
    #         anime_indices = self.df.loc[
    #             self.df['name'].isin(self.anime_pivot.index[indices.flatten()[1:]])
    #         ].index
    #     except Exception as e:
    #         return f"Error in fetching recommendations: {e}"
    
    #     # Return recommendations as a DataFrame
    #     return pd.DataFrame({
    #         'Anime Name': self.df.loc[anime_indices, 'name'].values,
    #         'Image URL': self.df.loc[anime_indices, 'image url'].values,
    #         'Genres': self.df.loc[anime_indices, 'genres'].values,
    #         'Rating': self.df.loc[anime_indices, 'average_rating'].values
    #     })
 
    
    # def get_user_based_recommendations(self, user_id, n_recommendations=5):
    #     if not self.knn_user_based:
    #         self.train_knn_user_based()
    #     user_id = float(user_id)
    #     if user_id not in self.user_pivot.index:
    #         return f"User '{user_id}' not found in the dataset."
    #     user_idx = self.user_pivot.index.get_loc(user_id)
    #     distances, indices = self.knn_user_based.kneighbors(
    #         self.user_pivot.iloc[user_idx, :].values.reshape(1, -1), 
    #         n_neighbors=n_recommendations + 1
    #     )
    #     user_rated_anime = set(self.user_pivot.columns[self.user_pivot.iloc[user_idx, :] > 0])
    #     all_neighbor_ratings = []
    #     for i in range(1, len(distances.flatten())):
    #         neighbor_idx = indices.flatten()[i]
    #         neighbor_rated_anime = self.user_pivot.iloc[neighbor_idx, :]
    #         neighbor_ratings = neighbor_rated_anime[neighbor_rated_anime > 0]
    #         all_neighbor_ratings.extend(neighbor_ratings.index)
    #     anime_counter = Counter(all_neighbor_ratings)
    #     recommendations = [(anime, count) for anime, count in anime_counter.items() if anime not in user_rated_anime]
    #     recommendations.sort(key=lambda x: x[1], reverse=True)
    #     top_anime_names = [rec[0] for rec in recommendations[:n_recommendations]]
    #     anime_indices = self.df[self.df['name'].isin(top_anime_names)].index

    #     return pd.DataFrame({
    #         'Anime name': self.df['name'].iloc[anime_indices].values,
    #         'Image URL': self.df['image url'].iloc[anime_indices].values,
    #         'Genres': self.df['genres'].iloc[anime_indices].values,
    #         'Rating': self.df['rating'].iloc[anime_indices].values
    #     })
    
    # def get_svd_recommendations(self, user_id, n=10):
    #     if not self.svd:
    #         self.train_svd()
    #     anime_ids = self.df['anime_id'].unique()
    #     predictions = [(anime_id, self.svd.predict(user_id, anime_id).est) for anime_id in anime_ids]
    #     predictions.sort(key=lambda x: x[1], reverse=True)
    #     top_anime_ids = [pred[0] for pred in predictions[:n]]
    #     anime_indices = self.df[self.df['anime_id'].isin(top_anime_ids)].index

    #     return pd.DataFrame({
    #         'Anime name': self.df['name'].iloc[anime_indices].values,
    #         'Image URL': self.df['image url'].iloc[anime_indices].values,
    #         'Genres': self.df['genres'].iloc[anime_indices].values,
    #         'Rating': self.df['average_rating'].iloc[anime_indices].values
    #     })

     # def get_item_based_recommendations(self, anime_name, n_recommendations=5):
    #     if not self.knn_item_based:
    #         self.train_knn_item_based()
    #     if anime_name not in self.anime_pivot.index:
    #         return f"Anime title '{anime_name}' not found in the dataset."
    #     query_index = self.anime_pivot.index.get_loc(anime_name)
    #     distances, indices = self.knn_item_based.kneighbors(
    #         self.anime_pivot.iloc[query_index, :].values.reshape(1, -1), 
    #         n_neighbors=n_recommendations + 1
    #     )
    #     anime_indices = [self.df[self.df['name'] == self.anime_pivot.index[idx]].index[0] for idx in indices.flatten()[1:]]

    #     return pd.DataFrame({
    #         'Anime name': self.df['name'].iloc[anime_indices].values,
    #         'Image URL': self.df['image url'].iloc[anime_indices].values,
    #         'Genres': self.df['genres'].iloc[anime_indices].values,
    #         'Rating': self.df['average_rating'].iloc[anime_indices].values
    #     })




    # def get_item_based_recommendations(self, anime_name, n_recommendations=5):
    #     if not self.knn_item_based:
    #         self.train_knn_item_based()
    #     if anime_name not in self.anime_pivot.index:
    #         return f"Anime title '{anime_name}' not found in the dataset."
    #     query_index = self.anime_pivot.index.get_loc(anime_name)
    #     distances, indices = self.knn_item_based.kneighbors(
    #         self.anime_pivot.iloc[query_index, :].values.reshape(1, -1), 
    #         n_neighbors=n_recommendations + 1
    #     )
    #     recommendations = [
    #         (self.anime_pivot.index[indices.flatten()[i]], distances.flatten()[i])
    #         for i in range(1, len(distances.flatten()))
    #     ]
    #     return recommendations

    
    # def get_svd_recommendations(self, user_id, n=10):
    #     if not self.svd:
    #         self.train_svd()
    #     anime_ids = self.df['anime_id'].unique()
    #     predictions = [(anime_id, self.svd.predict(user_id, anime_id).est) for anime_id in anime_ids]
    #     predictions.sort(key=lambda x: x[1], reverse=True)
    #     top_anime_ids = [pred[0] for pred in predictions[:n]]
    #     anime_names = set(self.df[self.df['anime_id'].isin(top_anime_ids)]['name'].tolist())
    #     return anime_names



    # def get_user_based_recommendations(self, user_id, n_recommendations=5):
    #     if not self.knn_user_based:
    #         self.train_knn_user_based()
    #     user_id = float(user_id)
    #     if user_id not in self.user_pivot.index:
    #         return f"User '{user_id}' not found in the dataset."
    #     user_idx = self.user_pivot.index.get_loc(user_id)
    #     distances, indices = self.knn_user_based.kneighbors(
    #         self.user_pivot.iloc[user_idx, :].values.reshape(1, -1), 
    #         n_neighbors=n_recommendations + 1
    #     )
    #     user_rated_anime = set(self.user_pivot.columns[self.user_pivot.iloc[user_idx, :] > 0])
    #     all_neighbor_ratings = []
    #     for i in range(1, len(distances.flatten())):
    #         neighbor_idx = indices.flatten()[i]
    #         neighbor_rated_anime = self.user_pivot.iloc[neighbor_idx, :]
    #         neighbor_ratings = neighbor_rated_anime[neighbor_rated_anime > 0]
    #         all_neighbor_ratings.extend(neighbor_ratings.index)
    #     anime_counter = Counter(all_neighbor_ratings)
    #     recommendations = [(anime, count) for anime, count in anime_counter.items() if anime not in user_rated_anime]
    #     recommendations.sort(key=lambda x: x[1], reverse=True)
    #     return recommendations[:n_recommendations]