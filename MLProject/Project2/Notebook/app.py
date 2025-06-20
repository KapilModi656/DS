import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from tensorflow import keras
from datetime import datetime, timedelta
data={
    "apple":"AAPL",
    "google":"GOOGL",
    "amazon":"AMZN",
    "microsoft":"MSFT",
    "bitcoin":"BTC",
    "ethereum":"ETH",
    "meta":"META"
}
today=datetime.now().strftime("%Y-%m-%d")
date=st.date_input("Select a date", value=pd.to_datetime(today))
stock= st.selectbox("Select a stock", ["APPLE", "GOOGLE", "AMAZON", "MICROSOFT", "BITCOIN", "ETHEREUM","META"])
button=st.button("Predict")
if button:
    st.write(f"Selected date: {date}")
    st.write(f"Selected stock: {stock}")
    stock=stock.lower()
    # Load the model

    model = keras.models.load_model(f"models/price_model_{stock}.keras")
    scaler_features = joblib.load(f"models/price_feat_{stock}.pkl")
    scaler_seq = joblib.load(f"models/price_seq_{stock}.pkl")
    scaler_y= joblib.load(f"models/price_y_{stock}.pkl")
    # Create a dummy input for prediction
    start_date= date - pd.Timedelta(days=110)
    df=yf.download(data[stock],start=start_date,end=date)
    df.columns = df.columns.droplevel(1)  # Drops 'AAPL'
    df = df.reset_index()
    
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True, errors='ignore')
    df.columns.name = None
    df.interpolate(method='time', inplace=True)
    df['Close_transformed'] = np.log1p(df['Close'])
    for i in range(1,60):
        df[f'Close_t-{i}'] = (df['Close_transformed'].shift(i))
    
    
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_diff'] = ta.trend.macd_diff(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['BB_mavg'] = ta.volatility.bollinger_mavg(df['Close'], window=20)
    df['BB_width'] = ta.volatility.bollinger_wband(df['Close'],
                                                    window=20, window_dev=2)
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
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
    df['Cumulative_Returns'] = df['Cumulative_Returns'].fillna(0)
   
    df['target'] = df['Close_transformed'].shift(-1)
    df['RollingMean'] = df['Close'].rolling(window=20).mean()
    df['RollingStd'] = df['Close'].rolling(window=20).std()
    last_60=df.tail(60)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    import plotly.graph_objects as go
    # Create candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=last_60.index,
            open=last_60['Open'],
            high=last_60['High'],
            low=last_60['Low'],
            close=last_60['Close'],
            increasing=dict(line=dict(color='green', width=1), fillcolor='green'),
            decreasing=dict(line=dict(color='red', width=1), fillcolor='red'),
            name='Candlestick'
        )
    ])

    # Customize layout
    
    

    
    print(df.tail())
    last=df.iloc[-1]
    nan_columns = last[last.isna()].index.tolist()
    print(f"NaN found in columns: {nan_columns}") if nan_columns else print("No NaN in the last row.")
    
    print(df.columns)
    
    # Make a prediction
    trans_columns=['Close_transformed','Close_t-1', 'Close_t-2', 'Close_t-3', 'Close_t-4', 'Close_t-5', 'Close_t-6', 'Close_t-7', 'Close_t-8', 'Close_t-9',
               'Close_t-10', 'Close_t-11', 'Close_t-12', 'Close_t-13', 'Close_t-14', 'Close_t-15', 'Close_t-16', 'Close_t-17', 'Close_t-18', 'Close_t-19', 
               'Close_t-20', 'Close_t-21', 'Close_t-22', 'Close_t-23', 'Close_t-24', 'Close_t-25', 'Close_t-26', 'Close_t-27', 'Close_t-28',
               'Close_t-29', 'Close_t-30', 'Close_t-31', 'Close_t-32', 'Close_t-33', 'Close_t-34', 'Close_t-35', 'Close_t-36', 'Close_t-37',
               'Close_t-38', 'Close_t-39', 'Close_t-40', 'Close_t-41', 'Close_t-42', 'Close_t-43', 'Close_t-44', 'Close_t-45', 'Close_t-46', 'Close_t-47', 
               'Close_t-48', 'Close_t-49', 'Close_t-50', 'Close_t-51', 'Close_t-52', 'Close_t-53','Close_t-54', 'Close_t-55', 'Close_t-56', 'Close_t-57', 'Close_t-58', 'Close_t-59']
    
    
    # Check for NaN in the last row before prediction
    last_row = df.iloc[-1]
    nan_columns = last_row[last_row.isna()].index.tolist()
    # Remove 'target' from NaN check only if it exists in the list
    if 'target' in nan_columns:
        nan_columns.remove('target')
    if nan_columns:
        st.error(f"Cannot predict: NaN found in columns: {nan_columns}")
    else:
        y = df['target'].values
        X = df[trans_columns].values
        # Ensure X has the correct shape for scaler_seq
        if X.shape[1] != scaler_seq.n_features_in_:
            st.error(f"Shape mismatch: X has {X.shape[1]} features, but scaler_seq expects {scaler_seq.n_features_in_}.")
        else:
            X_2d = X  # shape (samples, timesteps)
            X_scaled = scaler_seq.transform(X_2d)
            X = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
              # Reshape for LSTM input
            drop_cols = trans_columns + ['Volume_transformed', 'Open_transformed', 'High_transformed', 'Low_transformed','Close']
            X_feat = df.drop(columns=drop_cols, errors='ignore').values
            X_feat = X_feat[-1].reshape(1, X_feat.shape[1])  # Reshape for dense input
            X_feat = scaler_features.transform(X_feat)
            X_last_seq = X[-1].reshape(1, 60, 1)
            # Get last sample for features input
            X_last_feat = X_feat[-1].reshape(1, X_feat.shape[1])
            # Predict using both inputs
            predicted = model.predict([X_last_seq, X_last_feat])
            predicted_price = scaler_y.inverse_transform(predicted.reshape(1,1))[0][0]
            # Inverse log transformation
            st.write(f"Predicted price for {stock} on {date}: {np.expm1(predicted_price)}")


    fig.update_layout(
        title=f'{stock} - Last 60 Days Candlestick Chart',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        dragmode='pan',
        hoverlabel=dict(bgcolor='rgba(0, 0, 0, 0.8)', font=dict(color='white')),
        title_x=0.5,
        title_y=0.95,
        title_font=dict(size=20, color='white'),
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.2)',
            zeroline=False,
            tickformat='%Y-%m-%d',
            tickangle=45,
            ticks='outside',
            tickfont=dict(size=10, color='white'),
            
            linewidth=1,
            showline=True
        ),
        yaxis=dict(
            autorange=True,
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.2)',
            zeroline=False,
            ticks='outside',
            tickfont=dict(size=10, color='white'),
            
            linewidth=1,
            showline=True,
              # Adjust y-axis range
        ),
        width=1400,  # Try increasing
        height=600,
        margin=dict(l=20, r=20, t=40, b=0),
        xaxis_rangeslider_visible=False
        
    )
    fig.update_xaxes(tickformatstops=[
        dict(dtickrange=[None, 1000*60*60*24*30], value="%b %d"),  # < 1 month
        dict(dtickrange=[1000*60*60*24*30, None], value="%b %Y")   # >= 1 month
    ])
    st.plotly_chart(fig, use_container_width=True)
    
    st.plotly_chart(fig, use_container_width=True)
