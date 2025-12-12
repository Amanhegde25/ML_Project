import sys
from src.obj_locations import obj_locations
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np  # type: ignore
import pandas as pd # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import OneHotEncoder,StandardScaler # type: ignore


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=obj_locations()

    def get_preprocessor_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            num_pipeline = Pipeline(steps = [
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )
            logging.info(f"Numerical columns: {numerical_columns}")

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]          
            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            #reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # logging.info("Read train and test data completed")
            
            #drop target column from train and test dataframe
            target_column_name = "math_score"
            train_input_df = train_df.drop(columns = [target_column_name],axis = 1)
            test_input_df = test_df.drop(columns = [target_column_name],axis = 1)
            train_target_df = train_df[target_column_name]
            test_target_df = test_df[target_column_name]
            logging.info(f"After dropping target column shapes: {train_input_df.shape}, {test_input_df.shape}")

            #creating preprocessing object
            logging.info( f"Applying preprocessing object" )
            preprocessing_obj = self.get_preprocessor_object()            
            train_input_arr = preprocessing_obj.fit_transform(train_input_df)
            test_input_arr = preprocessing_obj.transform(test_input_df)
            train_arr  =  np.c_[train_input_arr, np.array(train_target_df)]
            test_arr  =  np.c_[test_input_arr, np.array(test_target_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)