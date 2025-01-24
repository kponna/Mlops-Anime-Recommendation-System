import sys
from anime_recommender.loggers.logging import logging
from anime_recommender.exception.exception import AnimeRecommendorException
from anime_recommender.entity.config_entity import ContentBasedModelConfig
from anime_recommender.entity.artifact_entity import ContentBasedModelArtifact, DataIngestionArtifact
from anime_recommender.utils.main_utils.utils import load_csv_data
from anime_recommender.utils.ml_utils.models.content_filtering_models import ContentBasedRecommender
from anime_recommender.constant import *
class ContentBasedModelTrainer:
    """Class to train the model, track metrics, and save the trained model."""

    def __init__(self, content_based_model_trainer_config: ContentBasedModelConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.content_based_model_trainer_config = content_based_model_trainer_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise AnimeRecommendorException(e, sys)

    def initiate_model_trainer(self) -> ContentBasedModelArtifact:
        try:
            logging.info("Loading ingested data...")
            df = load_csv_data(self.data_ingestion_artifact.feature_store_anime_file_path)
            logging.info("Training ContentBasedRecommender model...")
            
            # Initialize and train the model
            recommender = ContentBasedRecommender(df=df )
            
            # Save the model (TF-IDF and cosine similarity matrix)
            recommender.save_model(self.content_based_model_trainer_config.cosine_similarity_model_file_path)
            logging.info("Model saved successfully.")
            
            logging.info("Loading saved model to get recommendations...")
            cosine_recommendations = recommender.get_rec_cosine(title="One Piece", model_path=self.content_based_model_trainer_config.cosine_similarity_model_file_path, n_recommendations=10)
            logging.info(f"Cosine similarity recommendations: {cosine_recommendations}")
            
            # Return artifact with saved model path
            content_model_trainer_artifact = ContentBasedModelArtifact(
                cosine_similarity_model_file_path=self.content_based_model_trainer_config.cosine_similarity_model_file_path
            )
            return content_model_trainer_artifact
        except Exception as e:
            raise AnimeRecommendorException(f"Error in ContentBasedModelTrainer: {str(e)}", sys)

# import sys 
# from anime_recommendor.loggers.logging import logging
# from anime_recommendor.exception.exception import AnimeRecommendorException
# from anime_recommendor.entity.config_entity import ContentBasedModelConfig
# from anime_recommendor.entity.artifact_entity import ContentBasedModelArtifact,DataIngestionArtifact
# from anime_recommendor.utils.main_utils.utils import load_csv_data, save_model,load_object
# from anime_recommendor.utils.ml_utils.models.content_filtering_models import ContentBasedRecommender  


# class ContentBasedModelTrainer:
#     """
#     Class to train the model, track metrics, and save the trained model.
#     """
#     def __init__(self, content_based_model_trainer_config: ContentBasedModelConfig, data_ingestion_artifact: DataIngestionArtifact):
         
#         try:
#             self.content_based_model_trainer_config = content_based_model_trainer_config
#             self.data_ingestion_artifact = data_ingestion_artifact
#         except Exception as e:
#             raise AnimeRecommendorException(e, sys) 

#     def initiate_model_trainer(self) -> ContentBasedModelArtifact:
#         try:
#             logging.info("Loading ingested data...")
#             df = load_csv_data(self.data_ingestion_artifact.feature_store_anime_file_path)
#             recommender = ContentBasedRecommender(df) 
#             save_model(recommender.tfv, self.content_based_model_trainer_config.cosine_similarity_model_file_path) 
#             logging.info("Saving the trained model...")
#             logging.info("Loading saved model to get recommendations...")
#             model = load_object(self.content_based_model_trainer_config.cosine_similarity_model_file_path)
#             logging.info("Getting recommendations using cosine similarity...")
#             cosine_recommendations = recommender.get_rec_cosine(title="One Piece", n_recommendations=10,model=model)
#             logging.info(f"Cosine similarity recommendations: {cosine_recommendations}") 
#             content_model_trainer_artifact = ContentBasedModelArtifact(
#                 cosine_similarity_model_file_path=self.content_based_model_trainer_config.cosine_similarity_model_file_path 
#             )
#             return content_model_trainer_artifact

#         except Exception as e:
#             raise AnimeRecommendorException(f"Error in ContentBasedModelTrainer: {str(e)}",sys) 

 

    # def initiate_model_trainer(self, method: str) -> ContentBasedModelArtifact: 
    #     try:
    #         if method not in ['tfv_cosine', 'cv_sigmoid']:
    #             raise ValueError("Method must be either 'tfv_cosine' or 'cv_sigmoid'.")

    #         logging.info("Loading ingested data...")
    #         df = load_csv_data(self.data_ingestion_artifact.feature_store_anime_file_path)
    #         recommender = ContentBasedRecommender(df)

    #         if method == 'cv_sigmoid':
    #             logging.info("Getting recommendations using CV sigmoid kernel...")
    #             recommender._initialize_cv()
    #             logging.info("Getting recommendations using CV sigmoid kernel...")
    #             recommendations = recommender.get_rec_sig("One Piece", n_recommendations=10)
    #             model_to_save = recommender.count_vectorizer
    #             model_file_path = self.content_based_model_trainer_config.sigmoid_model_file_path
    #         else:
    #             logging.info("Getting recommendations using TFV cosine similarity...")
    #             recommender._initialize_tfv()
    #             logging.info("Getting recommendations using TFV cosine similarity...")    
    #             recommendations = recommender.get_rec_cos("One Piece", n_recommendations=10)
    #             model_to_save = recommender.tfv
    #             model_file_path = self.content_based_model_trainer_config.cosine_similarity_model_file_path

    #         logging.info(f"Recommendations: {recommendations}")
    #         save_model(model_to_save, model_file_path)

    #         content_model_trainer_artifact = ContentBasedModelArtifact(
    #             model_file_path=model_file_path
    #         )
    #         return content_model_trainer_artifact

    #     except Exception as e:
    #         raise AnimeRecommendorException(f"Error in ContentBasedModelTrainer: {str(e)}")

 



 




        # def initiate_model_trainer(self, method: str = 'tfv_cosine') -> ContentBasedModelArtifact: 
        # try:
        #     if method not in ['tfv_cosine', 'cv_sigmoid']:
        #         raise ValueError("Method must be either 'cosine' or 'sigmoid'.")

        #     logging.info("Loading ingested data...")
        #     df = load_csv_data(self.data_ingestion_artifact.feature_store_anime_file_path)
        #     recommender = ContentBasedRecommender(df)

        #     if method == 'cv_sigmoid':
        #         logging.info("Getting recommendations using TFV cosine similarity...")
        #         recommendations = recommender.get_rec_sig("One Piece", n_recommendations=10)
        #         model_to_save = recommender.count_vectorizer
        #         model_file_path = self.content_based_model_trainer_config.sigmoid_model_file_path
        #     else:
        #         logging.info("Getting recommendations using CV sigmoid kernel...")
        #         recommendations = recommender.get_rec_cos("One Piece", n_recommendations=10)
        #         model_to_save = recommender.tfv
        #         model_file_path = self.content_based_model_trainer_config.cosine_similarity_model_file_path

        #     logging.info(f"Recommendations: {recommendations}")
        #     save_model(model_to_save, model_file_path)

        #     content_model_trainer_artifact = ContentBasedModelArtifact(
        #         model_file_path=model_file_path
        #     )
        #     return content_model_trainer_artifact

        # except Exception as e:
        #     raise AnimeRecommendorException(f"Error in ContentBasedModelTrainer: {str(e)}")


    