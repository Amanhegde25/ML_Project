import sys
from src.exception import CustomException
from src.pipeline.train_pipeline import train_pipeline

if __name__=="__main__":
    try:
        train_pipeline()
    
    except Exception as e:
        raise CustomException(e,sys)