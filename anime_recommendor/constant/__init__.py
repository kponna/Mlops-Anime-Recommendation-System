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
# MODEL_FILE_NAME = "model.keras"
# SAVED_MODEL_DIR = os.path.join("saved_models") 
# SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "AnimeData"
DATA_INGESTION_DATABASE_NAME: str = "ANIMEDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested" 