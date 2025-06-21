import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import tensorflow as tf
from src.utils import save_model, load_model, save_object, load_object
from sklearn.preprocessing import MinMaxScaler
from src.transformer import build_multi_output_model
from tensorflow.keras.callbacks import EarlyStopping
@dataclass
class ModelTrainerConfig:
    apple_model_path: str = os.path.join('Models', 'apple_model.keras')
    bitcoin_model_path: str = os.path.join('Models', 'bitcoin_model.keras')
    ethereum_model_path: str = os.path.join('Models', 'ethereum_model.keras')
    google_model_path: str = os.path.join('Models', 'google_model.keras')
    microsoft_model_path: str = os.path.join('Models', 'microsoft_model.keras')
    meta_model_path: str = os.path.join('Models', 'meta_model.keras')
    amazon_model_path: str = os.path.join('Models', 'amazon_model.keras')
    apple_sequence_path: str = os.path.join('artifacts', 'apple_sequence.pkl')
    bitcoin_sequence_path: str = os.path.join('artifacts', 'bitcoin_sequence.pkl')
    ethereum_sequence_path: str = os.path.join('artifacts', 'ethereum_sequence.pkl')
    google_sequence_path: str = os.path.join('artifacts', 'google_sequence.pkl')
    microsoft_sequence_path: str = os.path.join('artifacts', 'microsoft_sequence.pkl')
    meta_sequence_path: str = os.path.join('artifacts', 'meta_sequence.pkl')
    amazon_sequence_path: str = os.path.join('artifacts', 'amazon_sequence.pkl')
    apple_features_path: str = os.path.join('artifacts', 'apple_features.pkl')
    bitcoin_features_path: str = os.path.join('artifacts', 'bitcoin_features.pkl')
    ethereum_features_path: str = os.path.join('artifacts', 'ethereum_features.pkl')
    google_features_path: str = os.path.join('artifacts', 'google_features.pkl')
    microsoft_features_path: str = os.path.join('artifacts', 'microsoft_features.pkl')
    meta_features_path: str = os.path.join('artifacts', 'meta_features.pkl')
    amazon_features_path: str = os.path.join('artifacts', 'amazon_features.pkl')
    apple_y_path: str = os.path.join('artifacts', 'apple_y.pkl')
    bitcoin_y_path: str = os.path.join('artifacts', 'bitcoin_y.pkl')
    ethereum_y_path: str = os.path.join('artifacts', 'ethereum_y.pkl')
    google_y_path: str = os.path.join('artifacts', 'google_y.pkl')
    microsoft_y_path: str = os.path.join('artifacts', 'microsoft_y.pkl')
    meta_y_path: str = os.path.join('artifacts', 'meta_y.pkl')
    amazon_y_path: str = os.path.join('artifacts', 'amazon_y.pkl')
class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
        self.scaler_sequence = MinMaxScaler()
        self.scaler_features = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def train_model(self, paths):
        try:
            logging.info("Starting model training")
            for path in paths:
                logging.info(f"Processing file: {path}")
                os.makedirs(os.path.dirname(self.trainer_config.amazon_features_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.trainer_config.amazon_model_path), exist_ok=True)
                df = pd.read_csv(path)

               
                for col in ['target', 'Trend', 'Signal']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                
                df.dropna(subset=['target', 'Trend', 'Signal'], inplace=True)

                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)

                seq_columns = [f'Close_t-{i}' for i in range(1, 60)] + ['Close']
                drop_columns = seq_columns + ['target', 'Trend', 'Signal','Date']

                X_seq = df[seq_columns].values
                X_seq = self.scaler_sequence.fit_transform(X_seq)
                X_seq = X_seq.reshape((X_seq.shape[0], 60, 1))

                X_features = df.drop(columns=drop_columns, errors='ignore').values
                X_features = self.scaler_features.fit_transform(X_features)

                y_price = df['target'].values.reshape(-1, 1)
                y_price = self.scaler_y.fit_transform(y_price)

                y_trend = df['Trend'].values
                y_signal = df['Signal'].values
                y_trend = y_trend + 1
                y_signal = y_signal + 1
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                model = build_multi_output_model(seq_len=60, d_model=64, n_features=X_features.shape[1], num_heads=4)
                model.fit(
                    [X_seq, X_features],
                    {'price_output': y_price, 'trend_output': y_trend, 'signal_output': y_signal},
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping]
                )

                model_path = getattr(self.trainer_config, f"{os.path.basename(path).split('.')[0]}_model_path")
                save_model(model, model_path)

                sequence_path = getattr(self.trainer_config, f"{os.path.basename(path).split('.')[0]}_sequence_path")
                features_path = getattr(self.trainer_config, f"{os.path.basename(path).split('.')[0]}_features_path")
                y_path = getattr(self.trainer_config, f"{os.path.basename(path).split('.')[0]}_y_path")

                save_object(self.scaler_sequence, sequence_path)
                save_object(self.scaler_features, features_path)
                save_object(self.scaler_y, y_path)

            logging.info("Model training completed successfully")
            return {
                'apple_model_path': self.trainer_config.apple_model_path,
                'bitcoin_model_path': self.trainer_config.bitcoin_model_path,
                'ethereum_model_path': self.trainer_config.ethereum_model_path,
                'google_model_path': self.trainer_config.google_model_path,
                'microsoft_model_path': self.trainer_config.microsoft_model_path,
                'meta_model_path': self.trainer_config.meta_model_path,
                'amazon_model_path': self.trainer_config.amazon_model_path
            }

        except Exception as e:
            raise CustomException(e, sys)
