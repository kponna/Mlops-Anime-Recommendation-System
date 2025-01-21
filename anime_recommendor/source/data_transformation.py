import sys 
import numpy as np
import pandas as pd

from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException
from anime_recommendor.utils.main_utils import export_data_to_dataframe
from anime_recommendor.constant import *
from anime_recommendor.entity.config_entity import DataTransformationConfig
from anime_recommendor.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact
 
class DataTransformation:
    """
    Class for handling data transformation for energy generation models.
    """
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_transformation_config:DataTransformationConfig):
        """
        Initializes the DataTransformation class with the given data ingestion and configuration artifacts.
        
        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact containing ingested data paths.
            data_transformation_config (DataTransformationConfig): Configuration object for data transformation.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise AnimeRecommendorException(e,sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        """
        Reads data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The DataFrame containing the data from the CSV file. 
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise AnimeRecommendorException(e,sys)
    
    @staticmethod
    def merge_data(anime_df: pd.DataFrame, rating_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges the anime and rating DataFrames on 'anime_id'.

        Args:
            anime_df (pd.DataFrame): DataFrame containing anime information.
            rating_df (pd.DataFrame): DataFrame containing user rating information.

        Returns:
            pd.DataFrame: Merged DataFrame on 'anime_id'.
        """
        try:
            merged_df = pd.merge(rating_df, anime_df, on="anime_id", how="inner")
            logging.info(f"Shape of the Merged dataframe:{merged_df.shape}")
            logging.info(f"Column names: {merged_df.columns}") 
            return merged_df
        except Exception as e:
            raise AnimeRecommendorException(e, sys)

    @staticmethod
    def clean_filter_data(merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the merged DataFrame by replacing 'UNKNOWN' with NaN, filling NaN values with median and also filters the data.

        Args:
            merged_df (pd.DataFrame): Merged DataFrame to clean and filter.

        Returns:
            pd.DataFrame: Cleaned and Filtered DataFrame with NaN values handled.
        """
        try:
            merged_df['average_rating'].replace('UNKNOWN', np.nan)
            merged_df['average_rating'] = pd.to_numeric(merged_df['average_rating'], errors='coerce')
            merged_df['average_rating'].fillna(merged_df['average_rating'].median())
            merged_df = merged_df[merged_df['average_rating'] > 7]
            logging.info(f"Shape of the Merged dataframe:{merged_df.shape}")
            logging.info(f"Column names: {merged_df.columns}")
            logging.info(f"Preview of the merged DataFrame:\n{merged_df.head()}")
            return merged_df
        except Exception as e:
            raise AnimeRecommendorException(e, sys)
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        """
        Initiates the data transformation process by reading, transforming, and saving the data.

        Returns:
            DataTransformationArtifact: The artifact containing paths to the transformed data. 
        """
        logging.info("Entering initiate_data_transformation method of DataTransformation class.")
        try:  
            anime_df = DataTransformation.read_data(self.data_ingestion_artifact.feature_store_anime_file_path)
            rating_df = DataTransformation.read_data(self.data_ingestion_artifact.feature_store_userrating_file_path) 
            merged_df = DataTransformation.merge_data(anime_df, rating_df)
            transformed_df = DataTransformation.clean_filter_data(merged_df)

            export_data_to_dataframe(transformed_df, self.data_transformation_config.merged_file_path)
            data_transformation_artifact = DataTransformationArtifact( 
                merged_file_path=self.data_transformation_config.merged_file_path
                            )
            
            return data_transformation_artifact
        except Exception as e:
            raise AnimeRecommendorException(e,sys)