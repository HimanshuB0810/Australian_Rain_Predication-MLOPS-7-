import os
import joblib 
from sklearn.ensemble import RandomForestClassifier
from src.custom_exception import CustomException
from src.logger import get_logger
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

logger = get_logger(__name__)

class ModelTraining:
    
    def __init__(self,input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.model = RandomForestClassifier(random_state=42)
        self.X_test_processed = None
        self.X_train_resampled = None
        self.y_test = None
        self.y_train_resampled = None

        os.makedirs(self.output_path,exist_ok=True)

        logger.info("Model Training Initailized")

    def load_data(self):
        try:
            self.X_test_processed = joblib.load(os.path.join(self.input_path,"X_test_processed.pkl"))
            self.X_train_resampled = joblib.load(os.path.join(self.input_path,"X_train_resampled.pkl"))
            self.y_test = joblib.load(os.path.join(self.input_path,"y_test.pkl"))
            self.y_train_resampled = joblib.load(os.path.join(self.input_path,"y_train_resampled.pkl"))

            logger.info("Data Loaded Successfully")

        except Exception as e:
            logger.error("Error While Loading Data")
            raise CustomException("Failed to load Data",e)
        
    def train_model(self):
        try:
            self.model.fit(self.X_train_resampled,self.y_train_resampled)

            joblib.dump(self.model, os.path.join(self.output_path,"model.pkl"))

            logger.info("Training And Saving Model Done")

        except Exception as e:
            logger.error("Error While Training Model")
            raise CustomException("Failed to Train Model",e)
        
    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test_processed)

            accuracy = accuracy_score(self.y_test,y_pred)
            precision = precision_score(self.y_test,y_pred,average="weighted")
            recall = recall_score(self.y_test,y_pred,average="weighted")
            f1 = f1_score(self.y_test,y_pred,average="weighted")

            logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

            logger.info("Model Evaluation Done")

        
        except Exception as e:
            logger.error("Error While Evaluating Model")
            raise CustomException("Failed to Evaluate Model",e)
        
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()
        logger.info("Model Training Done")

if __name__=="__main__":
    model_trainer = ModelTraining(input_path="artifacts/processed",output_path="artifacts/models")
    model_trainer.run()

