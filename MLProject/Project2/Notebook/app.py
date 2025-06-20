import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta
from tensorflow import keras
from datetime import datetime, timedelta
import os
st.set_page_config(layout="wide", page_title="ML Dashboard", page_icon="📈")
print("Current working dir:", os.getcwd())
# --- Top navigation bar (professional, clean, user-friendly) ---
st.markdown(
    """
    <nav style="width:100vw;position:relative;left:calc(-50vw + 50%);background:#232526;padding:0.9em 0;box-shadow:0 2px 12px #2224;display:flex;align-items:center;justify-content:space-between;z-index:100;">
        <div style="display:flex;align-items:center;gap:1.2em;margin-left:2em;">
            <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" width="42" style="border-radius:50%;border:2px solid #ffd700;box-shadow:0 2px 8px #ffd70022;" alt="AI Logo" />
            <span style="font-size:1.7em;color:#ffd700;font-weight:700;letter-spacing:1.2px;">ML Dashboard</span>
            <span style="color:#00ffae;font-size:1.08em;margin-left:1em;">Stock & Crypto Price Predictor</span>
        </div>
        <div style="margin-right:2em;display:flex;gap:1.2em;">
            <a href="https://github.com/" target="_blank" style="color:#00ffae;font-size:1em;text-decoration:none;font-weight:500;" title="GitHub Repository">GitHub</a>
            <a href="https://icons8.com" target="_blank" style="color:#ffd700;font-size:1em;text-decoration:none;font-weight:500;" title="Icons8">Icons8</a>
        </div>
    </nav>
    """,
    unsafe_allow_html=True
)

# --- Instructions bar (friendly, minimal, accessible) ---
st.markdown(
    """
    <div style="background:linear-gradient(90deg,#ffd70022 0%,#00ffae22 100%);padding:0.8em 1.2em;border-radius:12px;margin:1.3em 0 1.3em 0;box-shadow:0 2px 8px 0 #00ffae11;display:flex;align-items:center;gap:1em;">
        <span style="color:#ffd700;font-weight:600;font-size:1.1em;" aria-label="Instructions">How to use:</span>
        <span style="color:#e0e0e0;font-size:1.05em;">
            <ol style="margin:0;padding-left:1.2em;">
                <li>Select asset</li>
                <li>Choose prediction type</li>
                <li>Click <span style="color:#ffd700;">Predict</span></li>
            </ol>
            <span style="color:#00ffae;">Tip:</span> Try different assets for best results!
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Enhanced Custom CSS for dashboard styling (professional, user-friendly) ---
st.markdown(
    """
    <style>
    html, body, .stApp {
        background: linear-gradient(135deg, #191c24 0%, #232526 100%) fixed;
        color: #fff;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
        background: rgba(30,44,70,0.07);
        border-radius: 14px;
        box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.10);
        margin-bottom: 2rem;
        max-width: 100vw;
    }
    .stButton>button {
        color: #fff;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border: none;
        border-radius: 8px;
        padding: 0.7em 2.2em;
        font-size: 1.08em;
        font-weight: 600;
        transition: background 0.3s, color 0.3s, box-shadow 0.3s;
        box-shadow: 0 2px 8px 0 #23252655;
        letter-spacing: 1px;
        margin-top: 1em;
        outline: none;
    }
    .stButton>button:focus {
        outline: 2px solid #ffd700;
        outline-offset: 2px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ffd700 0%, #00ffae 100%);
        color: #232526;
        box-shadow: 0 4px 16px 0 #ffd70055;
    }
    .stSelectbox, .stDateInput {
        background: #23272f !important;
        color: #fff !important;
        border-radius: 8px !important;
        font-size: 1.08em !important;
        border: 2px solid #00ffae !important;
        box-shadow: 0 2px 6px 0 #23252622;
        margin-bottom: 1em !important;
    }
    .stPlotlyChart {
        background: #23272f;
        border-radius: 12px;
        padding: 1em;
        box-shadow: 0 2px 12px 0 #1e3c7244;
        margin-bottom: 1.5em;
        border: 1.5px solid #ffd70022;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffd700;
        text-shadow: 0 2px 8px #ffd70022;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    ::-webkit-scrollbar {
        width: 10px;
        background: #232526;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e3c72;
        border-radius: 8px;
    }
    .prediction-card {
        transition: box-shadow 0.2s, transform 0.2s;
        animation: popIn 1.2s;
        border: 2px solid #00ffae;
        background: linear-gradient(120deg, #232526 0%, #414345 100%);
        box-shadow: 0 4px 16px 0 #00ffae88, 0 2px 8px 0 #ffd70044;
    }
    .prediction-card:hover {
        box-shadow: 0 8px 24px 0 #ffd70099, 0 2px 8px 0 #00ffae66;
        transform: scale(1.02);
    }
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.97);}
        100% { opacity: 1; transform: scale(1);}
    }
    @media (max-width: 900px) {
        .block-container { padding: 1rem !important; }
        .stPlotlyChart { padding: 0.3em !important; }
        .side-by-side { flex-direction: column !important; }
        .side-by-side > div { width: 100% !important; }
    }
    .side-by-side {
        display: flex;
        flex-direction: row;
        gap: 2em;
        width: 100%;
        align-items: flex-start;
        justify-content: center;
    }
    .side-by-side > div {
        flex: 1 1 0;
        min-width: 0;
    }
    .fade-in {
        animation: fadeIn 1.2s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px);}
        to { opacity: 1; transform: translateY(0);}
    }
    .footer {
        margin-top: 2em;
        text-align: center;
        color: #b0b0b0;
        font-size: 1em;
        letter-spacing: 1px;
        padding: 1em 0 0.5em 0;
        border-top: 1px solid #232526;
        animation: fadeIn 1.5s;
        background: linear-gradient(90deg,#232526 60%,#181818 100%);
    }
    .footer a {
        color: #ffd700;
        text-decoration: none;
        font-weight: 500;
    }
    .footer span.emoji {
        font-size: 1.1em;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Dashboard Header ---
st.markdown(
    """
    <div class="fade-in" style="text-align:center; margin-bottom:2em;">
        <h1 style="font-size:2em; color:#ffd700; letter-spacing:1.5px; margin-bottom:0.1em; text-shadow:0 2px 8px #ffd70022; font-family:'Segoe UI', 'Roboto', 'Arial', sans-serif;">
            <span style="vertical-align:middle;" aria-label="Chart">📈</span>
            <span style="color:#00ffae;">Stock</span> & <span style="color:#ffd700;">Crypto</span> Price Predictor
        </h1>
        <div style="color:#b0b0b0; font-size:1em; text-shadow:0 1px 6px #00ffae22;">
            Powered by <span style="color:#00ffae;">Deep Learning</span> &amp; <span style="color:#ffd700;">Technical Analysis</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

data = {
    "apple": "AAPL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "microsoft": "MSFT",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "meta": "META"
}
today = datetime.now().strftime("%Y-%m-%d")

# --- Input Section ---
with st.container():
    st.markdown(
        """
        <div class="fade-in" style="background:rgba(44,62,80,0.18);padding:1.5em 1.2em 1.2em 1.2em;border-radius:14px;margin-bottom:2em;box-shadow:0 2px 8px 0 #00ffae22;">
            <h3 style="color:#00ffae;margin-bottom:0.7em;text-shadow:0 2px 8px #00ffae22;font-family:'Segoe UI', 'Roboto', 'Arial', sans-serif;">
                Select Asset
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    predict = st.selectbox("Select Prediction type:", ["Today Closing Price", "Trend", "Signal"], key="prediction_type")
    stock = st.selectbox("Select a stock", ["APPLE", "GOOGLE", "AMAZON", "MICROSOFT", "BITCOIN", "ETHEREUM", "META"], key="stock_select")
    button = st.button("Predict", key="predict_button")

if button:
    st.markdown(
        f"""
        <div class="fade-in" style="margin-bottom:1.2em;display:flex;flex-wrap:wrap;gap:1.5em;align-items:center;">
            <span style="color:#ffd700;font-weight:600;">Selected prediction type:</span>
            <span style="color:#fff;">{predict}</span>
            <span style="color:#ffd700;font-weight:600;">Prediction for today:</span>
            <span style="color:#fff;">{today}</span>
            <span style="color:#ffd700;font-weight:600;">Selected asset:</span>
            <span style="color:#fff;">{stock}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    stock = stock.lower()
    # --- Loading spinner for prediction ---
    with st.spinner('Predicting...'):
        model = keras.models.load_model(f"MLProject/Project2/Notebook/models/price_model_{stock}.keras")
        scaler_features = joblib.load(f"MLProject/Project2/Notebook/models/price_feat_{stock}.pkl")
        scaler_seq = joblib.load(f"MLProject/Project2/Notebook/models/price_seq_{stock}.pkl")
        scaler_y = joblib.load(f"MLProject/Project2/Notebook/models/price_y_{stock}.pkl")
        # Use today's date for prediction
        start_date = pd.to_datetime(today) - pd.Timedelta(days=110)
        end_date = pd.to_datetime(today) + pd.Timedelta(days=1)
        df = yf.download(data[stock], start=start_date, end=end_date)
        if df.empty:
            st.error("No data returned from yfinance for the selected stock and date range.")
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            df.drop(['index'], axis=1, inplace=True, errors='ignore')
            df.columns.name = None
            df.interpolate(method='time', inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    if isinstance(df[col], pd.DataFrame) or (hasattr(df[col], 'shape') and len(df[col].shape) > 1 and df[col].shape[1] == 1):
                        df[col] = df[col].iloc[:, 0]
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

            # --- Prediction Section ---
            trans_columns=['Close_transformed','Close_t-1', 'Close_t-2', 'Close_t-3', 'Close_t-4', 'Close_t-5', 'Close_t-6', 'Close_t-7', 'Close_t-8', 'Close_t-9',
                        'Close_t-10', 'Close_t-11', 'Close_t-12', 'Close_t-13', 'Close_t-14', 'Close_t-15', 'Close_t-16', 'Close_t-17', 'Close_t-18', 'Close_t-19', 
                        'Close_t-20', 'Close_t-21', 'Close_t-22', 'Close_t-23', 'Close_t-24', 'Close_t-25', 'Close_t-26', 'Close_t-27', 'Close_t-28',
                        'Close_t-29', 'Close_t-30', 'Close_t-31', 'Close_t-32', 'Close_t-33', 'Close_t-34', 'Close_t-35', 'Close_t-36', 'Close_t-37',
                        'Close_t-38', 'Close_t-39', 'Close_t-40', 'Close_t-41', 'Close_t-42', 'Close_t-43', 'Close_t-44', 'Close_t-45', 'Close_t-46', 'Close_t-47', 
                        'Close_t-48', 'Close_t-49', 'Close_t-50', 'Close_t-51', 'Close_t-52', 'Close_t-53','Close_t-54', 'Close_t-55', 'Close_t-56', 'Close_t-57', 'Close_t-58', 'Close_t-59']
            last_row = df.iloc[-1]
            nan_columns = last_row[last_row.isna()].index.tolist()
            if 'target' in nan_columns:
                nan_columns.remove('target')
            if nan_columns:
                st.error(f"Cannot predict: NaN found in columns: {nan_columns}")
            else:
                y = df['target'].values
                X = df[trans_columns].values
                if X.shape[1] != scaler_seq.n_features_in_:
                    st.error(f"Shape mismatch: X has {X.shape[1]} features, but scaler_seq expects {scaler_seq.n_features_in_}.")
                else:
                    X_2d = X
                    X_scaled = scaler_seq.transform(X_2d)
                    X = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
                    drop_cols = trans_columns + ['target','Close']
                    X_feat = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore').values
                    X_feat = X_feat[-1].reshape(1, X_feat.shape[1])
                    X_feat = scaler_features.transform(X_feat)
                    X_last_seq = X[-1].reshape(1, 60, 1)
                    X_last_feat = X_feat[-1].reshape(1, X_feat.shape[1])
                    predicted = model.predict([X_last_seq, X_last_feat])
                    # --- Output selection logic ---
                    pred_price = scaler_y.inverse_transform(predicted[0])[0][0]
                    pred_price_val = round(float(np.expm1(pred_price)), 2)
                    pred_trend = np.argmax(predicted[1])-1
                    pred_signal = np.argmax(predicted[2])-1
                    # Decide what to display based on user selection
                    if predict == "Today Closing Price":
                        display_val = f"${pred_price_val}"
                        display_label = "Predicted Closing Price"
                    elif predict == "Trend":
                        if pred_trend == 1:
                            display_val = "Stock price will go up 🚀📈"
                        elif pred_trend == 0:
                            display_val = "Stock price will be stable 🤝😐"
                        else:
                            display_val = "Stock price will go down 📉🔻"
                        display_label = "Predicted Trend"
                    elif predict == "Signal":
                        if pred_signal == 1:
                            display_val = f"Best Time to Buy {stock} stock 🟢💹"
                        elif pred_signal == 0:
                            display_val = f"Hold the {stock} stock is purchased already 🟡✋"
                        else:
                            display_val = f"Best Time to Sell {stock} stock 🔴💸"
                        display_label = "Predicted Signal"
                    else:
                        display_val = "-"
                        display_label = "Prediction"
                    # --- Responsive side-by-side layout ---
                    st.markdown('<div class="side-by-side fade-in" style="align-items:stretch;">', unsafe_allow_html=True)
                    # Prediction card
                    st.markdown(
                        f"""
                        <div class="prediction-card" style="
                            background: linear-gradient(120deg, #232526 0%, #414345 100%);
                            border-radius: 26px;
                            box-shadow: 0 8px 32px 0 #00ffaecc, 0 2px 8px 0 #ffd70099;
                            padding: 2.8em 1.7em 2.2em 1.7em;
                            margin: 2.5em 0 2em 0;
                            border: 2.5px solid #00ffae;
                            position: relative;
                            overflow: hidden;
                            max-width: 600px;
                            margin-left: auto;
                            margin-right: auto;
                            flex: 1 1 0;
                        ">
                            <div style="
                                position: absolute;
                                top: -40px;
                                right: -40px;
                                width: 120px;
                                height: 120px;
                                background: radial-gradient(circle, #00ffae55 0%, #23252600 80%);
                                z-index: 0;
                            "></div>
                            <div style="
                                position: absolute;
                                bottom: -40px;
                                left: -40px;
                                width: 120px;
                                height: 120px;
                                background: radial-gradient(circle, #ffd70055 0%, #23252600 80%);
                                z-index: 0;
                            "></div>
                            <h2 style="
                                color: #fff;
                                text-align: center;
                                margin: 0 0 0.5em 0;
                                font-size: 2.5em;
                                letter-spacing: 1px;
                                z-index: 2;
                                position: relative;
                                text-shadow: 0 2px 16px #ffd70033;
                                font-family:'Segoe UI', 'Roboto', 'Arial', sans-serif;
                            ">
                                <span style="color:#ffd700;">{stock.upper()}</span> {display_label}
                            </h2>
                            <div style="
                                text-align: center;
                                font-size: 1.4em;
                                color: #e0e0e0;
                                margin-bottom: 0.7em;
                                z-index: 2;
                                position: relative;
                                text-shadow: 0 1px 8px #00ffae33;
                            ">
                                For <span style="color:#ffd700;">{today}</span>
                            </div>
                            <div style="
                                text-align: center;
                                font-size: 3.2em;
                                font-weight: bold;
                                color: #00ffae;
                                margin-bottom: 0.2em;
                                letter-spacing: 2px;
                                z-index: 2;
                                position: relative;
                                text-shadow: 0 2px 24px #00ffae99;
                                animation: popIn 1.2s;
                            ">
                                {display_val}
                            </div>
                            <div style="
                                text-align: center;
                                font-size: 1.15em;
                                color: #b0b0b0;
                                z-index: 2;
                                position: relative;
                                text-shadow: 0 1px 6px #ffd70033;
                                display: { 'block' if predict == 'Today Closing Price' else 'none' };
                            ">
                                {f"(USD)" if predict == "Today Closing Price" else ""}
                            </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Chart
                    fig = go.Figure(data=[
                        go.Candlestick(
                            x=last_60.index,
                            open=last_60['Open'],
                            high=last_60['High'],
                            low=last_60['Low'],
                            close=last_60['Close'],
                            increasing=dict(line=dict(color='#00ffae', width=2), fillcolor='#00ffae'),
                            decreasing=dict(line=dict(color="#ff2222", width=2), fillcolor='#ff2222'),
                            name='Candlestick'
                        )
                    ])
                    fig.update_layout(
                        title={
                            'text': f'{stock.upper()} - Last 60 Days Candlestick Chart',
                            'y':0.93,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': dict(size=24, color='#ffd700', family='Segoe UI,Roboto,Arial,sans-serif')
                        },
                        yaxis_title='Price (USD)',
                        xaxis_title='Date',
                        template='plotly_dark',
                        plot_bgcolor='rgba(24,24,24,0.95)',
                        paper_bgcolor='rgba(24,24,24,0.95)',
                        font=dict(color='#fff', family='Segoe UI,Roboto,Arial,sans-serif'),
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1,
                            font=dict(color='#ffd700', size=13)
                        ),
                        hovermode='x unified',
                        dragmode='pan',
                        hoverlabel=dict(
                            bgcolor='rgba(30,44,70,0.95)',
                            font=dict(color='#ffd700', size=13)
                        ),
                        xaxis=dict(
                            type='date',
                            rangeslider=dict(visible=False),
                            showgrid=True,
                            gridcolor='rgba(0,255,174,0.13)',
                            zeroline=False,
                            tickformat='%b %d',
                            tickangle=0,
                            ticks='outside',
                            tickfont=dict(size=12, color='#fff'),
                            linewidth=2,
                            showline=True,
                            linecolor='#ffd700'
                        ),
                        yaxis=dict(
                            autorange=True,
                            showgrid=True,
                            gridcolor='rgba(255,215,0,0.10)',
                            zeroline=False,
                            ticks='outside',
                            tickfont=dict(size=12, color='#fff'),
                            linewidth=2,
                            showline=True,
                            linecolor='#00ffae'
                        ),
                        margin=dict(l=0, r=0, t=40, b=0),  # Reduce margins for more space
                        xaxis_rangeslider_visible=False
                    )
                    fig.update_xaxes(tickformatstops=[
                        dict(dtickrange=[None, 1000*60*60*24*30], value="%b %d"),
                        dict(dtickrange=[1000*60*60*24*30, None], value="%b %Y")
                    ])
                    fig.update_traces(
                        selector=dict(type='candlestick'),
                        increasing_line_color='#00ffae',
                        decreasing_line_color='#ff2222',
                        increasing_fillcolor='rgba(0,255,174,0.18)',
                        decreasing_fillcolor='rgba(255,34,34,0.18)',
                        hoverinfo='all'
                    )
                    st.plotly_chart(fig, use_container_width=True, height=600)
                    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    """
    <footer class="footer" style="margin-top:2em;text-align:center;color:#b0b0b0;font-size:1em;letter-spacing:1px;padding:1em 0 0.5em 0;border-top:1px solid #232526;background:linear-gradient(90deg,#232526 60%,#181818 100%);">
        Made with <span class="emoji" style="color:#ffd700;">&#10084;</span> by
        <a href="https://github.com/" target="_blank" style="color:#ffd700;text-decoration:none;font-weight:500;">Your Team</a>
        &middot; Powered by <span style="color:#00ffae;">Streamlit</span>
        <br>
        <span style="font-size:0.95em;color:#888;">&copy; 2024 | All Rights Reserved</span>
    </footer>
    """,
    unsafe_allow_html=True
)