import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.obj_locations import obj_locations

import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

class DataIngestion:
    def __init__(self):
        self.ingestion_config=obj_locations()

    def initiate_data_ingestion(self):
        # logging.info("Entered the data ingestion")
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            #create directorys
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            #create raw data file
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(f"Train and test data shape: {train_set.shape}, {test_set.shape}")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        