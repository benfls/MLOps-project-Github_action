from src.data_processing import DataProcessing
from src.model_training import ModelTrainer

if __name__ == "__main__":
    data_processor = DataProcessing(file_path="artifacts/raw/data.csv")
    data_processor.run()

    model_trainer = ModelTrainer(model_path="artifacts/models")
    model_trainer.run()