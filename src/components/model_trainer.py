import os
import sys
import numpy as np
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mse, mae, rmse, r2

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

            model_scores = {}

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")

                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)

                _, _, _, r2 = self.evaluate_model(y_test, y_test_pred)
                model_scores[model_name] = r2

            best_model_name = max(model_scores, key=model_scores.get)
            best_model_score = model_scores[best_model_name]

            logging.info(f"Best model: {best_model_name} with R2 = {best_model_score}")

            if best_model_score < 0.60:
                raise CustomException("No suitable model found", sys)

            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
