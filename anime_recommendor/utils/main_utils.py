import os
import sys
import pandas as pd
from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException

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