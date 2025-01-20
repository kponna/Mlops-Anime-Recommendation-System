import os 
import sys
import pymongo
import numpy as np
import pandas as pd

from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException
from anime_recommendor.entity.config_entity import DataIngestionConfig
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_URI")

class DataIngestion:
    """
    Handles the ingestion of energy generation data, including fetching data from MongoDB,
    exporting it to a DataFrame, splitting it into train, validation, and test sets, 
    and saving the datasets as CSV files.
    """
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        """
        Initializes the DataIngestion class with configuration details.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration object for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AnimeRecommendorException(e,sys)
 