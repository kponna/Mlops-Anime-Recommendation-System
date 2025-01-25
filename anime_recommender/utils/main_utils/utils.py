import os
import sys
import pandas as pd
import joblib
from anime_recommender.loggers.logging import logging
from anime_recommender.exception.exception import AnimeRecommendorException
from anime_recommender.constant import *
def export_data_to_dataframe(dataframe: pd.DataFrame, file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Saving DataFrame to file: {file_path}")
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(file_path, index=False, header=True)
            logging.info(f"DataFrame saved successfully to {file_path}.")
            return dataframe
        except Exception as e:
            raise AnimeRecommendorException(e, sys)

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a NumPy array.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        np.array: Data from the CSV file as a NumPy array.
    """
    try:
        df = pd.read_csv(file_path)
        return df 
    except Exception as e:
        raise AnimeRecommendorException(e, sys) from e

def save_model(model: object,file_path: str ) -> None:
    try:
        logging.info("Entered the save_model method of Main utils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(model, file_obj)
        logging.info("Completed saving the model object.")
    except Exception as e:
        raise AnimeRecommendorException(e, sys) from e
    
def load_object(file_path:str)-> object:
    """
    Loads an object from a file using pickle.
    Args:
        file_path (str): Path to the file containing the object.
    Returns:
        object: Loaded object.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return joblib.load(file_obj)
    except Exception as e:
        raise AnimeRecommendorException(e,sys) from e
    

import pymongo
import numpy as np
database_name = DATA_INGESTION_DATABASE_NAME
collection_name = ANIMEUSERRATINGS_COLLECTION_NAME 
MONGO_DB_URL = os.getenv("MONGO_URI")
def fetch_data_from_mongodb(database_name:str ,collection_name: str) -> pd.DataFrame:
        try:
            logging.info(f"Fetching data from MongoDB collection: {collection_name}") 
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]
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