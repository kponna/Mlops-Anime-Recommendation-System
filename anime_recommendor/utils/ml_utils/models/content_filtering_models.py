import sys
import pandas as pd
from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity

class ContentBasedRecommender:
    """
    A content-based recommender system using TF-IDF Vectorizer and Cosine Similarity.
    """
    def __init__(self, df): 
        try:
            # Drop missing values from the DataFrame
            self.df = df.dropna()
            
            # Create a Series mapping anime names to their indices
            self.indices = pd.Series(self.df.index, index=self.df['name']).drop_duplicates()
            
            # Initialize and fit the TF-IDF Vectorizer on the 'genres' column
            self.tfv = TfidfVectorizer(
                min_df=3,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                ngram_range=(1, 3),
                stop_words='english'
            )
            self.tfv_matrix = self.tfv.fit_transform(self.df['genres'])
             
            self.cosine_sim = cosine_similarity(self.tfv_matrix, self.tfv_matrix)
        except Exception as e:
            raise AnimeRecommendorException(e, sys)

    def get_rec_cosine(self, title, n_recommendations=5):
        """
        Get recommendations based on cosine similarity for a given anime title. 
        """
        try:
            # Check if the title is in the indices
            if title not in self.indices.index:
                return f"Anime title '{title}' not found in the dataset."

            # Get the index of the given anime title
            idx = self.indices[title]
            
            # Compute the cosine similarity scores for the given anime
            cosinesim_scores = list(enumerate(self.cosine_sim[idx]))
            cosinesim_scores = sorted(cosinesim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations + 1]
            
            # Get the indices of the recommended anime
            anime_indices = [i[0] for i in cosinesim_scores]
 
            return pd.DataFrame({
                'Anime name': self.df['name'].iloc[anime_indices].values,
                'Rating': self.df['average_rating'].iloc[anime_indices].values
            })
        except Exception as e:
            raise AnimeRecommendorException(e, sys)
        











# class ContentBasedRecommender:
#     def __init__(self, df):
#         try:
#             self.df = df.fillna("")  # Replace null values with empty strings
#             self.indices = pd.Series(df.index, index=df['name']).drop_duplicates()

#             # Initialize vectorizers and compute similarity matrices
#             self._initialize_tfv()
#             self._initialize_cv()
#             logging.info("ContentBasedRecommender initialized successfully.")

#         except Exception as e:
#             logging.error("Error during ContentBasedRecommender initialization.")
#             raise AnimeRecommendorException(e, sys)

#     def _initialize_tfv(self):
#         """
#         Initializes the TF-IDF vectorizer and computes the sigmoid kernel similarity matrix.
#         """
#         try:
#             self.tfv = TfidfVectorizer(
#                 min_df=3,
#                 strip_accents='unicode',
#                 analyzer='word',
#                 token_pattern=r'\w{1,}',
#                 ngram_range=(1, 3),
#                 max_features=5000,
#                 stop_words='english'
#             )
#             self.tfv_matrix = self.tfv.fit_transform(self.df['genres'])
#             self.sig = sigmoid_kernel(self.tfv_matrix, self.tfv_matrix)
#             logging.info("TF-IDF vectorizer initialized successfully.")
#         except Exception as e:
#             logging.error("Error initializing TF-IDF vectorizer.")
#             raise AnimeRecommendorException(e, sys)

#     def _initialize_cv(self):
#         """
#         Initializes the CountVectorizer and computes the cosine similarity matrix.
#         """
#         try:
#             self.count_vectorizer = CountVectorizer(
#                 analyzer='word',
#                 token_pattern=r'\w{1,}',
#                 ngram_range=(1, 3),
#                 max_features=5000,
#                 stop_words='english'
#             )
#             self.count_matrix = self.count_vectorizer.fit_transform(self.df['genres'])
#             self.cosine_sim = cosine_similarity(self.count_matrix, self.count_matrix)
#             logging.info("CountVectorizer initialized successfully.")
#         except Exception as e:
#             logging.error("Error initializing CountVectorizer.")
#             raise AnimeRecommendorException(e, sys)

#     def get_rec_sig(self, title, n_recommendations=10):
#         """
#         Generates recommendations using the sigmoid kernel similarity.
#         """
#         try:
#             if title not in self.indices.index:
#                 logging.warning(f"Anime title '{title}' not found in the dataset.")
#                 return f"Anime title '{title}' not found in the dataset."

#             idx = self.indices[title]
#             sig_scores = list(enumerate(self.sig[idx]))
#             sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations + 1]
#             anime_indices = [i[0] for i in sig_scores]
#             logging.info(f"Generated sigmoid kernel recommendations for '{title}'.")
#             return pd.DataFrame({'Anime name': self.df['name'].iloc[anime_indices].values, 'Rating': self.df['average_rating'].iloc[anime_indices].values})
#         except Exception as e:
#             logging.error(f"Error generating sigmoid kernel recommendations for '{title}'.")
#             raise AnimeRecommendorException(e, sys)

#     def get_rec_cos(self, title, n_recommendations=10):
#         """
#         Generates recommendations using the cosine similarity.
#         """
#         try:
#             if title not in self.indices.index:
#                 logging.warning(f"Anime title '{title}' not found in the dataset.")
#                 return f"Anime title '{title}' not found in the dataset."

#             idx = self.indices[title]
#             sim_scores = list(enumerate(self.cosine_sim[idx]))
#             sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations + 1]
#             anime_indices = [i[0] for i in sim_scores]
#             logging.info(f"Generated cosine similarity recommendations for '{title}'.")
#             return pd.DataFrame({'Anime name': self.df['name'].iloc[anime_indices].values, 'Rating': self.df['average_rating'].iloc[anime_indices].values})
#         except Exception as e:
#             logging.error(f"Error generating Cosine similarity recommendations for '{title}'.")
#             raise AnimeRecommendorException(e, sys)















# import sys
# import pandas as pd
# from anime_recommendor.loggers.logging import logging
# from anime_recommendor.exception.exception import AnimeRecommendorException
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity
  
# class ContentBasedRecommender:
#     def __init__(self, df):
#         try: 
#             self.df = df
#             self.df = df.fillna("") 
#             self.indices = pd.Series(df.index, index=df['name']).drop_duplicates()

#             # TF-IDF Vectorizer and sigmoid kernel
#             self.tfv = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), max_features=5000,stop_words='english')
#             self.tfv_matrix = self.tfv.fit_transform(self.df['genres'])
#             self.sig = sigmoid_kernel(self.tfv_matrix, self.tfv_matrix)

#             # CountVectorizer and cosine similarity
#             self.count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words='english', max_features=5000)
#             self.count_matrix = self.count_vectorizer.fit_transform(self.df['genres'])
#             self.cosine_sim = cosine_similarity(self.count_matrix, self.count_matrix)
#             logging.info("ContentBasedRecommender initialized successfully.")

#         except Exception as e:
#             logging.error("Error during ContentBasedRecommender initialization.")
#             raise AnimeRecommendorException(e, sys)

#     def get_rec_sig(self, title, n_recommendations=10):
#         try:
#             if title not in self.indices.index:
#                 logging.warning(f"Anime title '{title}' not found in the dataset.")
#                 return f"Anime title '{title}' not found in the dataset."

#             idx = self.indices[title]
#             sig_scores = list(enumerate(self.sig[idx]))
#             sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations + 1]
#             anime_indices = [i[0] for i in sig_scores]
#             logging.info(f"Generated sigmoid kernel recommendations for '{title}'.")
#             return pd.DataFrame({'Anime name': self.df['name'].iloc[anime_indices].values, 'Rating': self.df['average_rating'].iloc[anime_indices].values})
#         except Exception as e:
#             logging.error(f"Error generating sigmoid kernel recommendations for '{title}'.")
#             raise AnimeRecommendorException(e, sys)
#     def get_rec_cos(self, title, n_recommendations=10):
#         try:
#             if title not in self.indices.index:
#                 logging.warning(f"Anime title '{title}' not found in the dataset.")
#                 return f"Anime title '{title}' not found in the dataset."

#             idx = self.indices[title]
#             sim_scores = list(enumerate(self.cosine_sim[idx]))
#             sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations + 1]
#             anime_indices = [i[0] for i in sim_scores]
#             logging.info(f"Generated cosine similarity recommendations for '{title}'.")
#             return pd.DataFrame({'Anime name': self.df['name'].iloc[anime_indices].values, 'Rating': self.df['average_rating'].iloc[anime_indices].values})
#         except Exception as e:
#             logging.error(f"Error generating Cosine similarity recommendations for '{title}'.")
#             raise AnimeRecommendorException(e, sys)

