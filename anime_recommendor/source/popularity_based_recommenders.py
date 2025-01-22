import sys
from anime_recommendor.exception.exception import AnimeRecommendorException
from anime_recommendor.loggers.logging import logging
from anime_recommendor.utils.main_utils.utils import load_csv_data 
from anime_recommendor.entity.artifact_entity import DataIngestionArtifact
from anime_recommendor.utils.ml_utils.models.popularity_based_filtering import PopularityBasedFiltering 


class PopularityBasedRecommendor:
    # def __init__(self,data_transformation_artifact = DataTransformationArtifact):
    #     try:
    #         self.data_transformation_artifact = data_transformation_artifact
    #     except Exception as e:
    #         raise AnimeRecommendorException(e,sys)

    def __init__(self,data_ingestion_artifact = DataIngestionArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise AnimeRecommendorException(e,sys)
        
    def initiate_model_trainer(self,filter_type:str):
        try:
            logging.info("Loading transformed data...")
            df = load_csv_data(self.data_ingestion_artifact.feature_store_anime_file_path)

            recommender = PopularityBasedFiltering(df)

            if filter_type == 'popular_animes': 
                popular_animes = recommender.popular_animes(n =10) 
                logging.info(f"Popular Anime recommendations: {popular_animes}") 

            elif filter_type == 'top_ranked_animes': 
                top_ranked_animes = recommender.top_ranked_animes(n =10) 
                logging.info(f"top_ranked_animes recommendations: {top_ranked_animes}")  

            elif filter_type == 'overall_top_rated_animes': 
                overall_top_rated_animes = recommender.overall_top_rated_animes(n =10) 
                logging.info(f"overall_top_rated_animes recommendations: {overall_top_rated_animes}") 

            elif filter_type == 'favorite_animes': 
                favorite_animes = recommender.favorite_animes(n =10) 
                logging.info(f"favorite_animes recommendations: {favorite_animes}") 

            elif filter_type == 'top_animes_members': 
                top_animes_members = recommender.top_animes_members(n = 10) 
                logging.info(f"top_animes_members recommendations: {top_animes_members}") 

            elif filter_type == 'popular_anime_among_members': 
                popular_anime_among_members = recommender.popular_anime_among_members(n =10) 
                logging.info(f"popular_anime_among_members recommendations: {popular_anime_among_members}") 
            
            elif filter_type == 'top_avg_rated': 
                top_avg_rated = recommender.top_avg_rated(n =10) 
                logging.info(f"top_avg_rated recommendations: {top_avg_rated}") 

        except Exception as e:
            raise AnimeRecommendorException(e,sys)
        
# if __name__ =="__main__":

#     training_pipeline_config = TrainingPipelineConfig()  
#     data_ingestion_config = DataIngestionConfig(training_pipeline_config)
#     data_ingestion = DataIngestion(data_ingestion_config)
#     logging.info("Initiating Data Ingestion.") 
#     data_ingestion_artifact = data_ingestion.ingest_data()
#     logging.info(f"Data ingestion completed.")
#     print(data_ingestion_artifact)

#     # Data Transformation
#     data_transformation_config = DataTransformationConfig(training_pipeline_config)
#     data_transformation = DataTransformation(data_ingestion_artifact,data_transformation_config)
#     logging.info("Initiating Data Transformation.")
#     data_transformation_artifact = data_transformation.initiate_data_transformation()
#     logging.info("Data Transformation Completed.")
#     print(data_transformation_artifact)
    
#     popularity_recommendations = PopularityBasedFiltering(data_transformation_artifact=data_transformation_artifact)
#     recommendations = popularity_recommendations.initiate_model_trainer(filter_type='top_avg_rated')
#     print(recommendations)