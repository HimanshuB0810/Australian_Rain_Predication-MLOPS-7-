from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import joblib

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self,input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.preprocessor = None
        self.df = None

        os.makedirs(self.output_path,exist_ok=True)
        logger.info("Data Processing Started")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data Loaded Successfully")
        except Exception as e:
            logger.error("Error while data loading")
            raise CustomException("Error while loading data",e)
        
    def preprocess(self):
        try:
            self.df['Date']=pd.to_datetime(self.df['Date'])
            self.df['Year']=self.df['Date'].dt.year
            self.df['Month']=self.df['Date'].dt.month
            self.df['Day']=self.df['Date'].dt.day

            self.df.drop(columns=['Date'],inplace=True)

            cat_cols = self.df.select_dtypes(include="str").columns
            num_cols = self.df.select_dtypes(exclude="str").columns

            for col in num_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())

            self.df.dropna(inplace=True)
            logger.info("Data Processing Done")

            return num_cols

        except Exception as e:
            logger.error("Error while data loading")
            raise CustomException("Error while loading data",e)
        
    def ohe(self,num_cols):
        try:
            cat_cols=['Location','WindGustDir','WindDir9am','WindDir3pm','RainToday']
            self.preprocessor = ColumnTransformer(
            transformers=[
                ("num",StandardScaler(),num_cols),
                ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
            ]
        )   
            logger.info("Ohe Done")
            return self.preprocessor
        
        except Exception as e:
            logger.error("Error while Ohe")
            raise CustomException("Error while Ohe",e)
        
    def split_data_and_Smote(self):
        try:
            X = self.df.drop(columns=['RainTomorrow'])
            y = self.df['RainTomorrow']

            y = y.replace({
            "No": 0,
            "Yes": 1
            })

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)

            smote = SMOTE(random_state=42)

            X_train_resampled,y_train_resampled = smote.fit_resample(X_train_processed,y_train)

            joblib.dump(X_train_resampled,os.path.join(self.output_path,"X_train_resampled.pkl"))
            joblib.dump(y_train_resampled,os.path.join(self.output_path,"y_train_resampled.pkl"))

            joblib.dump(X_test_processed,os.path.join(self.output_path,"X_test_processed.pkl"))
            joblib.dump(y_test,os.path.join(self.output_path,"y_test.pkl"))

            logger.info("Data Spiltting and Smote Done")

        except Exception as e:
            logger.error("Error while Data Spiltting and Smote")
            raise CustomException("Error while Data Spiltting and Smote",e)

    def run(self):
        self.load_data()
        num_cols=self.preprocess()
        self.ohe(num_cols)
        self.split_data_and_Smote()

        logger.info("Data Processing Done")

if __name__=="__main__":
    data_processor = DataProcessing(input_path="artifacts/raw/data.csv",output_path="artifacts/processed")
    data_processor.run()

