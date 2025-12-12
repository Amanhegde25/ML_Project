from dataclasses import dataclass
import os

@dataclass
class obj_locations:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    trained_model_file_path=os.path.join("artifacts","model.pkl")