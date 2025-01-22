from dataclasses import dataclass
from typing import Optional
@dataclass
class DataIngestionArtifact: 
    feature_store_anime_file_path:str
    feature_store_userrating_file_path:str
 
@dataclass
class DataTransformationArtifact:
    merged_file_path:str

@dataclass
class CollaborativeModelArtifact:
    svd_file_path: Optional[str] = None
    item_based_knn_file_path: Optional[str] = None
    user_based_knn_file_path: Optional[str] = None

@dataclass
class ContentBasedModelArtifact:
    cosine_similarity_model_file_path:str