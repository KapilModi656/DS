from dataclasses import dataclass
from datetime import datetime
import yfinance as yf
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    tickers= ['AAPL','MSFT','META','GOOGL','BTC-USD','ETH-USD','AMZN']
    apple_path:str=os.path.join('Data','apple.csv')
    bitcoin_path:str=os.path.join('Data','bitcoin.csv')
    ethereum_path:str=os.path.join('Data','ethereum.csv')
    google_path:str=os.path.join('Data','google.csv')
    microsoft_path:str=os.path.join('Data','microsoft.csv')
    meta_path:str=os.path.join('Data','meta.csv')
    amazon_path:str=os.path.join('Data','amazon.csv')

    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('Stated Ingestion component')
        try:
            data=yf.download(self.ingestion_config.tickers,group_by='ticker',period='max',auto_adjust=True)
            os.makedirs(os.path.dirname(self.ingestion_config.apple_path),exist_ok=True)
            apple = data['AAPL'].copy()
            bitcoin = data['BTC-USD'].copy()
            meta = data['META'].copy()
            google = data['GOOGL'].copy()
            amazon = data['AMZN'].copy()
            microsoft = data['MSFT'].copy()
            ethereum = data['ETH-USD'].copy()

            # Reset index if needed
            apple.reset_index(inplace=True)
            bitcoin.reset_index(inplace=True)
            meta.reset_index(inplace=True)
            google.reset_index(inplace=True)
            amazon.reset_index(inplace=True)
            microsoft.reset_index(inplace=True)
            ethereum.reset_index(inplace=True)
            asset=[apple,bitcoin,meta,google,amazon,microsoft,ethereum]
            for df in asset:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                df.columns.name=None
            for df in asset:
                first_valid=df['Close'].first_valid_index()
                df.drop(df.index[:df.index.get_loc(first_valid)],inplace=True)
            apple.to_csv(self.ingestion_config.apple_path)
            bitcoin.to_csv(self.ingestion_config.bitcoin_path)
            meta.to_csv(self.ingestion_config.meta_path)
            google.to_csv(self.ingestion_config.google_path)
            amazon.to_csv(self.ingestion_config.amazon_path)
            microsoft.to_csv(self.ingestion_config.microsoft_path)
            ethereum.to_csv(self.ingestion_config.ethereum_path)
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.apple_path,
                self.ingestion_config.bitcoin_path,
                self.ingestion_config.meta_path,
                self.ingestion_config.google_path,
                self.ingestion_config.amazon_path,
                self.ingestion_config.microsoft_path,
                self.ingestion_config.ethereum_path
            )
        except Exception as e:
            raise CustomException(e,sys)
       

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    paths = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.transform_data(paths)
    model_trainer = ModelTrainer()
    model_path=model_trainer.train_model(paths)

