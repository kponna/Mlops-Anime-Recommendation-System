import sys 
from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException
from anime_recommendor.entity.config_entity import ContentBasedModelConfig
from anime_recommendor.entity.artifact_entity import ContentBasedModelArtifact,DataIngestionArtifact
from anime_recommendor.utils.main_utils.utils import load_csv_data, save_model
from anime_recommendor.utils.ml_utils.models.content_filtering_models import ContentBasedRecommender


class ContentBasedModelTrainer:
    """
    Class to train the model, track metrics, and save the trained model.
    """
    def __init__(self, content_based_model_trainer_config: ContentBasedModelConfig, data_ingestion_artifact: DataIngestionArtifact):
         
        try:
            self.content_based_model_trainer_config = content_based_model_trainer_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise AnimeRecommendorException(e, sys)
        
    def initiate_model_trainer(self) -> ContentBasedModelArtifact:
        try:
            logging.info("Loading transformed data...")
            df = load_csv_data(self.data_ingestion_artifact.feature_store_anime_file_path)
            recommender = ContentBasedRecommender(df) 

            logging.info("Getting recommendations using cosine similarity...")
            cosine_recommendations = recommender.get_rec_cosine("One Piece", n_recommendations=10)
            logging.info(f"Cosine similarity recommendations: {cosine_recommendations}")

            save_model(recommender.tfv, self.content_based_model_trainer_config.tfidf_model_file_path) 

            content_model_trainer_artifact = ContentBasedModelArtifact(
                tfidf_model_file_path=self.content_based_model_trainer_config.tfidf_model_file_path 
            )
            return content_model_trainer_artifact

        except Exception as e:
            raise AnimeRecommendorException(f"Error in ContentBasedModelTrainer: {str(e)}")
