import os
import sys
import pandas as pd
import joblib
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

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a NumPy array.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        np.array: Data from the CSV file as a NumPy array.
    """
    try:
        df = pd.read_csv(file_path)
        return df 
    except Exception as e:
        raise AnimeRecommendorException(e, sys) from e

def save_model(model: object,file_path: str ) -> None:
    try:
        logging.info("Entered the save_model method of Main utils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(model, file_obj)
        logging.info("Completed saving the model object.")
    except Exception as e:
        raise AnimeRecommendorException(e, sys) from e