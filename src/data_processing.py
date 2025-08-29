import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import os
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.processed_data_path = "artifacts/processed"
        os.makedirs(self.processed_data_path, exist_ok=True)

        logger.info("DataProcessing instance created.")

    def load_data(self):
        """Load data from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully from {self.file_path}")
        except Exception as e:
            logger.error(f"Error loading data from {self.file_path}: {e}")
            raise CustomException("Failled to read data", e)
        
    def handle_outliers(self, column: str):
        try:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_value = Q1 - 1.5 * IQR
            Upper_value = Q3 + 1.5 * IQR

            sepal_median = np.median(self.data[column])

            for i in self.data[column]:
                if i < lower_value or i > Upper_value:
                    self.data[column] = self.data[column].replace(i, sepal_median)

            logger.info(f"Outliers in {column} handled successfully.")

        except Exception as e:
            logger.error(f"Error while handling outliers : {e}")
            raise CustomException("Failled to handle outliers", e)
        
    def split_data(self):
        try:
            X = self.data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
            y = self.data["Species"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logger.info("Data splited successfully")

            joblib.dump(X_train, os.path.join(self.processed_data_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.processed_data_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.processed_data_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.processed_data_path, "y_test.pkl"))

            logger.info("Filed saved successfully")

        except Exception as e:
            logger.error(f"Error while splitting data : {e}")
            raise CustomException("Failled to split data", e)
        
    def run(self):
        try:
            self.load_data()
            self.handle_outliers("SepalLengthCm")
            self.split_data()
            logger.info("Data processing completed successfully.")
        
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise CustomException("Failled to process data", e)

if __name__ == "__main__":
    data_processor = DataProcessing(file_path="artifacts/raw/data.csv")
    data_processor.run()