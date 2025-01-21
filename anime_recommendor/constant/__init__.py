import os
  
"""
Defining common constant variables for training pipeline
"""
PIPELINE_NAME: str = "AnimeRecommendor"
ARTIFACT_DIR: str = "Artifacts"
ANIME_FILE_NAME: str = "Animes.csv"
RATING_FILE_NAME:str = "UserRatings.csv"
MERGED_FILE_NAME:str = "Anime_UserRatings.csv"
ZIP_FILE_PATH:str = 'datasets/archive.zip'
DATASETS_FILE_PATH:str = "datasets"


ANIME_COLLECTION_NAME:str = "Animes"
RATING_COLLECTION_NAME:str = "UserRatings"
 
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
""" 
DATA_INGESTION_DATABASE_NAME: str = "ANIMEDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"

"""
Data Transformation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_TRANSFORMATION_DIR:str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = "transformed" 
