import os
import sys
import numpy as np
import pandas as pd
import pymongo
from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException
from anime_recommendor.entity.config_entity import DataIngestionConfig
from anime_recommendor.entity.artifact_entity import DataIngestionArtifact
from anime_recommendor.utils.main_utils import export_data_to_dataframe
MONGO_DB_URL = os.getenv("MONGO_URI")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AnimeRecommendorException(e, sys)
    
    def fetch_data_from_mongodb(self, collection_name: str) -> pd.DataFrame:
        try:
            logging.info(f"Fetching data from MongoDB collection: {collection_name}")
            database_name = self.data_ingestion_config.database_name 
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1) 
            df.replace({"na":np.nan},inplace=True)
            logging.info(f"Shape of the dataframe:{df.shape}")
            logging.info(f"Column names: {df.columns}")
            logging.info(f"Preview of the DataFrame:\n{df.head()}")
            logging.info("Data fetched successfully from MongoDB.")
            return df
        except pymongo.errors.ServerSelectionTimeoutError as e:
            logging.error("Could not connect to MongoDB. Please check if MongoDB is running and the connection URI is correct.")
            raise AnimeRecommendorException(e, sys)
        except Exception as e:
            raise AnimeRecommendorException(e, sys)
    
    def ingest_data(self) -> DataIngestionArtifact:
        try:
            anime_df = self.fetch_data_from_mongodb(self.data_ingestion_config.anime_collection_name)
            rating_df = self.fetch_data_from_mongodb(self.data_ingestion_config.rating_collection_name) 

            export_data_to_dataframe(anime_df, file_path=self.data_ingestion_config.feature_store_anime_file_path)
            export_data_to_dataframe(rating_df, file_path=self.data_ingestion_config.feature_store_userrating_file_path)
 
            dataingestionartifact = DataIngestionArtifact(feature_store_anime_file_path=self.data_ingestion_config.feature_store_anime_file_path,
                                                          feature_store_userrating_file_path=self.data_ingestion_config.feature_store_userrating_file_path)
            
            return dataingestionartifact
        except Exception as e:
            raise AnimeRecommendorException(e, sys) 