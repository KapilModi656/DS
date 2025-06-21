import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
from src.utils import save_object



class DataTransformation:
    

    def transform_data(self, paths, sequence_length: int = 60):
        try:
            for path in paths:
                logging.info(f"Starting data transformation for {path}")
                df=pd.read_csv(path)
                logging.info("Starting data transformation")
                df.fillna(method='ffill',inplace=True)
                df.fillna(method='bfill',inplace=True)
                df['SMA_20']=ta.trend.sma_indicator(df['Close'],window=20)
                df['SMA_50']=ta.trend.sma_indicator(df['Close'],window=50)
                df['EMA_20']=ta.trend.ema_indicator(df['Close'],window=20)
                df['EMA_50']=ta.trend.ema_indicator(df['Close'],window=50)
                df['RSI']=ta.momentum.rsi(df['Close'],window=14)
                df['MACD']=ta.trend.macd(df['Close'])
                df['MACD_signal']=ta.trend.macd_signal(df['Close'])
                df['MACD_diff']=ta.trend.macd_diff(df['Close'])
                df['ATR']=ta.volatility.average_true_range(df['High'],df['Low'],df['Close'],window=13)
                df['BB_upper']=ta.volatility.bollinger_hband(df['Close'],window=20,window_dev=2)
                df['BB_lower']=ta.volatility.bollinger_lband(df['Close'],window=20,window_dev=2)
                df['BB_mavg']=ta.volatility.bollinger_mavg(df['Close'],window=20)
                df['BB_width']=ta.volatility.bollinger_wband(df['Close'],window=20,window_dev=2)
                df['CCI']=ta.trend.cci(df['High'],df['Low'],df['Close'],window=20)
                df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
                df['ADX_pos'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14)
                df['ADX_neg'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14)
                df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                # Calculate Chaikin Oscillator manually
                acc_dist = ta.volume.AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).acc_dist_index()
                df['Chaikin_AO'] = acc_dist.ewm(span=3, adjust=False).mean() - acc_dist.ewm(span=10, adjust=False).mean()
                kst_indicator = ta.trend.KSTIndicator(
                    close=df['Close'],
                    window1=10, window2=15, window3=20, window4=30,
                    roc1=10, roc2=15, roc3=20, roc4=30
                )
                df['KST'] = kst_indicator.kst()
                df['KST_signal'] = kst_indicator.kst_sig()
                df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
                df['Ultimate_Oscillator'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'], window1=7, window2=14, window3=28)
                df['TRIX'] = ta.trend.trix(df['Close'], window=15)
                df['ROC'] = ta.momentum.roc(df['Close'], window=12)
                df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
                vortex = ta.trend.VortexIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
                df['Vortex_Pos'] = vortex.vortex_indicator_pos()
                df['Vortex_Neg'] = vortex.vortex_indicator_neg()
                df['Force_Index'] = ta.volume.force_index(df['Close'], df['Volume'], window=13)
                df['Price_Change'] = df['Close'].pct_change()
                df['Price_Change'] = df['Price_Change'].fillna(0)
                
            
                
                df['Cumulative_Returns'] = (1 + df['Price_Change']).cumprod() - 1
                
                df['Trend'] = df['Close'].diff()
                df['Trend'] = df['Trend'].apply(lambda x: 1 if x > 0.00 else (-1 if x < 0.00 else 0))
                df['Signal'] = 0  
                df.loc[(df['RSI'] < 30) & (df['Close'] < df['SMA_20']), 'Signal'] = 1 
                df.loc[(df['RSI'] > 70) & (df['Close'] > df['SMA_20']), 'Signal'] = -1  
                for i in range(1,60):
                    df[f"Close_t-{i}"]=df['Close'].shift(i)
                
                df['target']=df['Close'].shift(-1)
                df['Rolling_Mean']= df['Close'].rolling(window=20).mean()
                df['Rolling_std']= df['Close'].rolling(window=20).std()
                df.dropna(inplace=True)
                df.to_csv(path, index=False)
            logging.info("Feature engineering completed successfully")

            logging.info("Data scaling completed successfully")

            logging.info("Data saving completed successfully")

            logging.info("Data transformation completed successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
