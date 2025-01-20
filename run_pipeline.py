import sys

from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException

from anime_recommendor.source.data_ingestion import DataIngestion
from anime_recommendor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
 
if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()  
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating Data Ingestion.") 
        data_ingestion_artifact = data_ingestion.ingest_data()
        logging.info(f"Data ingestion completed.")
        print(data_ingestion_artifact)
    except Exception as e:
            raise AnimeRecommendorException(e, sys)