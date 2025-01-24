import sys
from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException
from anime_recommendor.entity.config_entity import CollaborativeModelConfig
from anime_recommendor.entity.artifact_entity import DataTransformationArtifact, CollaborativeModelArtifact
from anime_recommendor.utils.main_utils.utils import load_csv_data, save_model, load_object
from anime_recommendor.utils.ml_utils.models.collaborative_filtering_models import CollaborativeAnimeRecommender

class CollaborativeModelTrainer:
    """
    Class to train the model, track metrics, and save the trained model.
    """
    def __init__(self, collaborative_model_trainer_config: CollaborativeModelConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.collaborative_model_trainer_config = collaborative_model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise AnimeRecommendorException(e, sys)

    def initiate_model_trainer(self, model_type: str) -> CollaborativeModelArtifact:
        try:
            logging.info("Loading transformed data...")
            df = load_csv_data(self.data_transformation_artifact.merged_file_path)
            recommender = CollaborativeAnimeRecommender(df)

            if model_type == 'svd':
                logging.info("Training and saving SVD model...")
                recommender.train_svd()
                save_model(recommender.svd, self.collaborative_model_trainer_config.svd_trained_model_file_path)

                logging.info("Loading pre-trained SVD model...")
                svd_model = load_object(self.collaborative_model_trainer_config.svd_trained_model_file_path)
                svd_recommendations = recommender.get_svd_recommendations(user_id=375, n=10, svd_model=svd_model)
                logging.info(f"SVD recommendations: {svd_recommendations}")
                return CollaborativeModelArtifact(
                    svd_file_path=self.collaborative_model_trainer_config.svd_trained_model_file_path
                )

            elif model_type == 'item_knn':
                logging.info("Training and saving KNN item-based model...")
                recommender.train_knn_item_based()
                save_model(recommender.knn_item_based, self.collaborative_model_trainer_config.item_knn_trained_model_file_path)

                logging.info("Loading pre-trained item-based KNN model...")
                item_knn_model = load_object(self.collaborative_model_trainer_config.item_knn_trained_model_file_path)
                item_based_recommendations = recommender.get_item_based_recommendations(
                    anime_name='One Piece', n_recommendations=10, knn_item_model=item_knn_model
                )
                logging.info(f"Item Based recommendations: {item_based_recommendations}")
                return CollaborativeModelArtifact(
                    item_based_knn_file_path=self.collaborative_model_trainer_config.item_knn_trained_model_file_path
                )

            elif model_type == 'user_knn':
                logging.info("Training and saving KNN user-based model...")
                recommender.train_knn_user_based()
                save_model(recommender.knn_user_based, self.collaborative_model_trainer_config.user_knn_trained_model_file_path)

                logging.info("Loading pre-trained user-based KNN model...")
                user_knn_model = load_object(self.collaborative_model_trainer_config.user_knn_trained_model_file_path)
                user_based_recommendations = recommender.get_user_based_recommendations(
                    user_id=817, n_recommendations=10, knn_user_model=user_knn_model
                )
                logging.info(f"User Based recommendations: {user_based_recommendations}")
                return CollaborativeModelArtifact(
                    user_based_knn_file_path=self.collaborative_model_trainer_config.user_knn_trained_model_file_path
                )

            else:
                raise ValueError("Invalid model_type. Choose from 'svd', 'item_knn', or 'user_knn'.")

        except Exception as e:
            raise AnimeRecommendorException(f"Error in CollaborativeModelTrainer: {str(e)}", sys)
        
        
        
        
        
        
        
        
        
        
        
        # import sys 
# from anime_recommendor.loggers.logging import logging
# from anime_recommendor.exception.exception import AnimeRecommendorException
# from anime_recommendor.entity.config_entity import CollaborativeModelConfig
# from anime_recommendor.entity.artifact_entity import DataTransformationArtifact, CollaborativeModelArtifact
# from anime_recommendor.utils.main_utils.utils import load_csv_data,save_model,load_object
# from anime_recommendor.utils.ml_utils.models.collaborative_filtering_models import CollaborativeAnimeRecommender


# class CollaborativeModelTrainer:
#     """
#     Class to train the model, track metrics, and save the trained model.
#     """
#     def __init__(self, collaborative_model_trainer_config: CollaborativeModelConfig, data_transformation_artifact: DataTransformationArtifact):
         
#         try:
#             self.collaborative_model_trainer_config = collaborative_model_trainer_config
#             self.data_transformation_artifact = data_transformation_artifact
#         except Exception as e:
#             raise AnimeRecommendorException(e, sys)  
     

#     def initiate_model_trainer(self, model_type: str) -> CollaborativeModelArtifact:
#         try:
#             logging.info("Loading transformed data...")
#             df = load_csv_data(self.data_transformation_artifact.merged_file_path)
#             recommender = CollaborativeAnimeRecommender(df)

#             if model_type == 'svd':
#                 logging.info("Training and saving SVD model...")
#                 train_svd = recommender.train_svd() 
#                 save_model(recommender.svd, self.collaborative_model_trainer_config.svd_trained_model_file_path)

#                 logging.info("Loading pre-trained SVD model...")
#                 svd_model = load_object(self.collaborative_model_trainer_config.svd_trained_model_file_path)
#                 svd_recommendations = recommender.get_svd_recommendations(user_id=375, n=10, svd_model=svd_model)
#                 logging.info(f"SVD recommendations: {svd_recommendations}")
#                 return CollaborativeModelArtifact(
#                     svd_file_path=self.collaborative_model_trainer_config.svd_trained_model_file_path
#                 )

#             elif model_type == 'item_knn':
#                 logging.info("Training and saving KNN item-based model...")
#                 recommender.train_knn_item_based() 
#                 save_model(recommender.knn_item_based, self.collaborative_model_trainer_config.item_knn_trained_model_file_path)

#                 logging.info("Loading pre-trained item-based KNN model...")
#                 item_knn_model = load_object(self.collaborative_model_trainer_config.item_knn_trained_model_file_path)
#                 item_based_recommendations = recommender.get_item_based_recommendations(
#                     anime_name='One Piece', n_recommendations=10, item_knn_model=item_knn_model
#                 )
#                 logging.info(f"Item Based recommendations: {item_based_recommendations}") 
#                 return CollaborativeModelArtifact(
#                     item_based_knn_file_path=self.collaborative_model_trainer_config.item_knn_trained_model_file_path
#                 )

#             elif model_type == 'user_knn':
#                 logging.info("Training and saving KNN user-based model...")
#                 recommender.train_knn_user_based()
#                 save_model(recommender.knn_user_based, self.collaborative_model_trainer_config.user_knn_trained_model_file_path)
#                 logging.info("Loading pre-trained user-based KNN model...")
#                 user_knn_model = load_object(self.collaborative_model_trainer_config.user_knn_trained_model_file_path)
#                 user_based_recommendations = recommender.get_user_based_recommendations(
#                     user_id=817, n_recommendations=10, user_knn_model=user_knn_model
#                 )
#                 logging.info(f"User Based recommendations: {user_based_recommendations}") 
#                 return CollaborativeModelArtifact(
#                     user_based_knn_file_path=self.collaborative_model_trainer_config.user_knn_trained_model_file_path
#                 ) 
#             else:
#                 raise ValueError("Invalid model_type. Choose from 'svd', 'item_knn', or 'user_knn'.")

#         except Exception as e:
#             raise AnimeRecommendorException(f"Error in CollaborativeModelTrainer: {str(e)}")
        
        
        
        
# import sys 
# from anime_recommendor.loggers.logging import logging
# from anime_recommendor.exception.exception import AnimeRecommendorException
# from anime_recommendor.entity.config_entity import CollaborativeModelConfig
# from anime_recommendor.entity.artifact_entity import DataTransformationArtifact, CollaborativeModelArtifact
# from anime_recommendor.utils.main_utils.utils import load_csv_data,save_model
# from anime_recommendor.utils.ml_utils.models.collaborative_filtering_models import CollaborativeAnimeRecommender


# class CollaborativeModelTrainer:
#     """
#     Class to train the model, track metrics, and save the trained model.
#     """
#     def __init__(self, collaborative_model_trainer_config: CollaborativeModelConfig, data_transformation_artifact: DataTransformationArtifact):
         
#         try:
#             self.collaborative_model_trainer_config = collaborative_model_trainer_config
#             self.data_transformation_artifact = data_transformation_artifact
#         except Exception as e:
#             raise AnimeRecommendorException(e, sys)  
#     def initiate_model_trainer(self, model_type: str) -> CollaborativeModelArtifact:
#         try:
#             logging.info("Loading transformed data...")
#             df = load_csv_data(self.data_transformation_artifact.merged_file_path)
#             recommender = CollaborativeAnimeRecommender(df)

#             if model_type == 'svd':
#                 logging.info("Training and saving SVD model...")
#                 recommender.train_svd()
#                 svd_recommendations = recommender.get_svd_recommendations(user_id=375, n=10)
#                 logging.info(f"SVD recommendations: {svd_recommendations}")
#                 save_model(recommender.svd, self.collaborative_model_trainer_config.svd_trained_model_file_path)
#                 return CollaborativeModelArtifact(
#                     svd_file_path=self.collaborative_model_trainer_config.svd_trained_model_file_path
#                 )

#             elif model_type == 'item_knn':
#                 logging.info("Training and saving KNN item-based model...")
#                 recommender.train_knn_item_based()
#                 item_based_recommendations = recommender.get_item_based_recommendations(anime_name='One Piece', n_recommendations=10)
#                 logging.info(f"Item Based recommendations: {item_based_recommendations}")
#                 save_model(recommender.knn_item_based, self.collaborative_model_trainer_config.item_knn_trained_model_file_path)
#                 return CollaborativeModelArtifact(
#                     item_based_knn_file_path=self.collaborative_model_trainer_config.item_knn_trained_model_file_path
#                 )

#             elif model_type == 'user_knn':
#                 logging.info("Training and saving KNN user-based model...")
#                 recommender.train_knn_user_based()
#                 user_based_recommendations = recommender.get_user_based_recommendations(user_id=817, n_recommendations=10)
#                 logging.info(f"User Based recommendations: {user_based_recommendations}")
#                 save_model(recommender.knn_user_based, self.collaborative_model_trainer_config.user_knn_trained_model_file_path)
#                 return CollaborativeModelArtifact(
#                     user_based_knn_file_path=self.collaborative_model_trainer_config.user_knn_trained_model_file_path
#                 )

#             else:
#                 raise ValueError("Invalid model_type. Choose from 'svd', 'item_knn', or 'user_knn'.")

#         except Exception as e:
#             raise AnimeRecommendorException(f"Error in CollaborativeModelTrainer: {str(e)}")