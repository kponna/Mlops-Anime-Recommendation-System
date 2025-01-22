import sys

from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException

from anime_recommendor.source.data_ingestion import DataIngestion
from anime_recommendor.source.data_transformation import DataTransformation
from anime_recommendor.source.collaborative_recommenders import CollaborativeModelTrainer
from anime_recommendor.source.content_based_recommenders import ContentBasedModelTrainer
from anime_recommendor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataTransformationConfig,CollaborativeModelConfig ,ContentBasedModelConfig
 
if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()  
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating Data Ingestion.") 
        data_ingestion_artifact = data_ingestion.ingest_data()
        logging.info(f"Data ingestion completed.")
        print(data_ingestion_artifact)

        # Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_ingestion_artifact,data_transformation_config)
        logging.info("Initiating Data Transformation.")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed.")
        print(data_transformation_artifact)

        # Collaborative Model Training
        collaborative_model_trainer_config = CollaborativeModelConfig(training_pipeline_config)
        collaborative_model_trainer = CollaborativeModelTrainer(collaborative_model_trainer_config= collaborative_model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        logging.info("Initiating Collaborative Model training.")
        collaborative_model_trainer_artifact = collaborative_model_trainer.initiate_model_trainer()
        logging.info("Collaborative Model training completed.")
        print(collaborative_model_trainer_artifact)

        # Content Based Model Training
        content_based_model_trainer_config = ContentBasedModelConfig(training_pipeline_config)
        content_based_model_trainer = ContentBasedModelTrainer(content_based_model_trainer_config=content_based_model_trainer_config,data_ingestion_artifact=data_ingestion_artifact)
        logging.info("Initiating Content Based Model training.")
        content_based_model_trainer_artifact = content_based_model_trainer.initiate_model_trainer()
        logging.info("Content Based Model training completed.")
        print(content_based_model_trainer_artifact)

    except Exception as e:
            raise AnimeRecommendorException(e, sys)