import zipfile
import os
import sys
import json
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv
from anime_recommendor.loggers.logging import logging
from anime_recommendor.exception.exception import AnimeRecommendorException
from anime_recommendor.constant import * 

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
 
# Load environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_URI")

class NetworkDataExtract:
    def __init__(self):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
        except Exception as e:
            raise AnimeRecommendorException(e, sys)
        
    def extract_csv_files(self, zip_file_path, destination_folder):
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                file_names = zip_ref.namelist()
                csv_files = [file for file in file_names if file.endswith('.csv')]
                for file in csv_files:
                    file_name = os.path.basename(file)
                    destination_path = os.path.join(destination_folder, file_name)
                    with open(destination_path, 'wb') as f:
                        f.write(zip_ref.read(file))
            print("CSV files extracted successfully.")
            return [os.path.join(destination_folder, os.path.basename(file)) for file in csv_files]
        except Exception as e:
            raise AnimeRecommendorException(e, sys)
        
    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise AnimeRecommendorException(e, sys)
        
    def insert_data_mongodb(self, records, database, collection_name):
        try:
            db = self.mongo_client[database]
            col = db[collection_name]
            col.insert_many(records)
            return len(records)
        except Exception as e:
            raise AnimeRecommendorException(e, sys)

if __name__ == '__main__':
    zip_file_path = ZIP_FILE_PATH
    destination_folder = DATASETS_FILE_PATH
    database = DATA_INGESTION_DATABASE_NAME 
    network_obj = NetworkDataExtract()
    
    # Extract CSV files
    csv_files = network_obj.extract_csv_files(zip_file_path, destination_folder)
    
    for csv_file in csv_files:
        collection_name = os.path.splitext(os.path.basename(csv_file))[0] 
        records = network_obj.csv_to_json_convertor(file_path=csv_file) 
        # Insert records into MongoDB under appropriate collection
        no_of_records = network_obj.insert_data_mongodb(records, database, collection_name)
        print(f"Inserted {no_of_records} records from {csv_file} into MongoDB collection '{collection_name}'.")