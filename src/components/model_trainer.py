import os
import sys
import numpy as np
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_set, test_set):
        try:
            logging.info("Model training started")

            X_train, X_test, y_train, y_test = (
                train_set[:, :-1],
                test_set[:, :-1],
                train_set[:, -1],
                test_set[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "Decision Tree": DecisionTreeRegressor(),
                "SVR": SVR(),
                "AdaBoost": AdaBoostRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Lasso": {},
                "Ridge": {},
                "SVR": {},
            }
            best_models = {}

            for model_name, model in models.items():
                logging.info(f"Tuning {model_name}")

                param_grid = params[model_name]

                if param_grid:
                    gs = GridSearchCV(
                        model,
                        param_grid,
                        cv=3,
                        scoring="r2",
                        n_jobs=-1
                    )
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                y_test_pred = best_model.predict(X_test)
                r2 = r2_score(y_test, y_test_pred)

                best_models[model_name] = {
                    "model": best_model,
                    "r2": r2
                }

            best_model_name = max(best_models, key=lambda x: best_models[x]["r2"])
            best_model_score = best_models[best_model_name]["r2"]
            best_model = best_models[best_model_name]["model"]

            if best_model_score < 0.60:
                raise CustomException("No suitable model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                f"Best model: {best_model_name} with R2 score: {best_model_score}"
            )

            return best_model, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
