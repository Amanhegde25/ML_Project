import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.base import clone # type: ignore

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        model_names = list(models.keys())
        model_objs = list(models.values())

        for i in range(len(model_objs)):
            name = model_names[i]
            model = model_objs[i]
            param_grid = param.get(name, {})

            logging.info(f"{name}")
            if param_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                    error_score='raise'
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                logging.info(f"Best params: {gs.best_params_}")

            else:
                best_model = clone(model)
                best_model.fit(X_train, y_train)

            if not hasattr(best_model, "n_features_in_"):
                best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score
            logging.info(f"Train Score: {train_model_score} Test Score: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models_notuning(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            name = list(models.keys())[i]
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[name] = test_model_score
            logging.info(f"{name}: Train Score: {train_model_score} Test Score: {test_model_score}")
        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)