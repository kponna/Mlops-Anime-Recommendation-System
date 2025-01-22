import sys
import numpy as np
import pandas as pd
from anime_recommendor.loggers.logging import logging  
from anime_recommendor.exception.exception import AnimeRecommendorException

class PopularityBasedFiltering:
    def __init__(self,df):
        try:
            self.df = df
            self.df['average_rating'] = pd.to_numeric(self.df['average_rating'], errors='coerce')
            self.df['average_rating'].fillna(self.df['average_rating'].median())
        except Exception as e:
            raise AnimeRecommendorException(e,sys)

    def popular_animes(self,n=10):
        sorted_df = self.df.sort_values(by=['popularity'], ascending=[ True])
        top_n_anime = sorted_df.head(n)
        return top_n_anime[['name',  'popularity']]
    
    def top_ranked_animes(self,n=10):
        self.df['rank'] = self.df['rank'].replace('UNKNOWN', np.nan).astype(float)
        df_filtered = self.df[self.df['rank'] > 1]
        sorted_df = df_filtered.sort_values(by=['rank'], ascending=True)
        top_n_anime = sorted_df.head(n)
        return top_n_anime[['name', 'rank']] 
    
    def overall_top_rated_animes(self,n = 10):
        sorted_df = self.df.sort_values(by=['average_rating'], ascending=False)
        top_n_anime = sorted_df.head(n)
        return top_n_anime[['name', 'average_rating']] 
    
    def favorite_animes(self,n=10):
        sorted_df = self.df.sort_values(by=['favorites'], ascending=False)
        top_n_anime = sorted_df.head(n)
        return top_n_anime[['name', 'favorites']] 
    
    def top_animes_members(self,n=10):
        sorted_df = self.df.sort_values(by=['members'], ascending=False)
        top_n_anime = sorted_df.head(n)
        return top_n_anime[['name', 'members']]
    
    def popular_anime_among_members(self,n=10):
        sorted_df = self.df.sort_values(by=['members', 'average_rating'], ascending=[False, False]).drop_duplicates(subset='name')['name'] 
        popular_animes= sorted_df.head(n) 
        return popular_animes
    
    def top_avg_rated(self,n=10): 
        self.df['average_rating'] = pd.to_numeric(self.df['average_rating'], errors='coerce')
        
        # Replace NaN values with the median
        # median_rating = self.df['average_rating'].median()
        # self.df['average_rating'].fillna(median_rating )
        # Select top N animes by average rating
        top_animes = (
            self.df.drop_duplicates(subset='name')
                    .nlargest(n, 'average_rating')[['name', 'average_rating']]
        )  
        return top_animes