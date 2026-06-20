from src.model_training import ModelTraining
from src.data_processing import DataProcessing

if __name__=="__main__":
    data_processor = DataProcessing(input_path="artifacts/raw/data.csv",output_path="artifacts/processed")
    data_processor.run()

    model_trainer = ModelTraining(input_path="artifacts/processed",output_path="artifacts/models")
    model_trainer.run()