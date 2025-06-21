import os
import sys
import pandas as pd
from dataclasses import dataclass, field
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.logger import logging
from src.exception import CustomException
from src.Components.Data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.utils import save_object, load_object, load_model
import numpy as np
@dataclass
class PredictionPipelineConfig:
    data_ingestion_config: DataIngestion = DataIngestion()
    data_transformation_config: DataTransformation = DataTransformation()
    tickers: dict = field(default_factory=lambda: {
        'AAPL': 'apple',
        'BTC-USD': 'bitcoin',
        'ETH-USD': 'ethereum',
        'GOOGL': 'google',
        'MSFT': 'microsoft',
        'META': 'meta',
        'AMZN': 'amazon'
    })
    sequence_length: int = 60
class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()

    def run(self, ticker_name: str):
        try:
            logging.info("Starting prediction pipeline")

            # Step 1: Data Ingestion
            paths = self.config.data_ingestion_config.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            DataTransformation = self.config.data_transformation_config
            logging.info("Starting data transformation")
            DataTransformation.transform_data(paths, sequence_length=self.config.sequence_length)
            logging.info("Data transformation completed")
            
            # Validate paths
            if ticker_name not in self.config.tickers:
                raise CustomException(f"Ticker name {ticker_name} is not valid.", sys)

            for path in paths:
                if self.config.tickers[ticker_name] in path:
                    df_path = path
                    break
            else:
                raise CustomException(f"Data for ticker {ticker_name} not found in the provided paths.", sys)
            

            df = pd.read_csv(df_path)

            # Validate DataFrame columns
            
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            # Step 2: Data Transformation
            seq_columns = [f'Close_t-{i}' for i in range(1, 60)] + ['Close']
            drop_columns = seq_columns + ['target', 'Trend', 'Signal', 'Date']

            X_seq = df[seq_columns].values
            X_feat = df.drop(columns=drop_columns, errors='ignore').values

            sequence_path = os.path.join('artifacts', f"{self.config.tickers[ticker_name]}_sequence.pkl")
            features_path = os.path.join('artifacts', f"{self.config.tickers[ticker_name]}_features.pkl")

            sequence = load_object(sequence_path)
            features = load_object(features_path)

            X_seq = sequence.transform(X_seq)
            X_feat = features.transform(X_feat)
            X_last_seq = X_seq[-1].reshape(1, 60, 1)
            X_last_feat = X_feat[-1].reshape(1, X_feat.shape[1])
            
            logging.info(f"Sequence and features loaded from {sequence_path} and {features_path}")

            # Step 3: Model Prediction
            model_path = os.path.join('Models', f"{self.config.tickers[ticker_name]}_model.keras")
            model = load_model(model_path)

            scaler_y_path = os.path.join('artifacts', f"{self.config.tickers[ticker_name]}_y.pkl")
            scaler_y = load_object(scaler_y_path)

            logging.info(f"Model loaded from {model_path}")

            predictions = model.predict([X_last_seq, X_last_feat])
            logging.info("Model prediction completed")

            # Step 4: Post-processing
            y_price = scaler_y.inverse_transform(predictions[0].reshape(-1, 1))[0][0]
            y_trend = np.argmax(predictions[1])
            y_signal = np.argmax(predictions[2])

            logging.info(f"Predicted Price: {y_price}, Trend: {y_trend}, Signal: {y_signal}")

            return {
                'price': y_price,
                'trend': y_trend,
                'signal': y_signal
            }

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    pipeline = PredictionPipeline()
    ticker_name = 'AAPL'  # Example ticker name
    result = pipeline.run(ticker_name)
    print(f"Prediction Result for {ticker_name}: {result}")