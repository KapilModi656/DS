import plotly.graph_objects as go
import numpy as np

# ...existing code...

# Ensure data is sorted by date in ascending order
last_60 = last_60.sort_index()

# Drop rows with NaN or zero values in OHLC columns
last_60 = last_60.dropna(subset=['Open', 'High', 'Low', 'Close'])
last_60 = last_60[(last_60[['Open', 'High', 'Low', 'Close']] != 0).all(axis=1)]

# Remove extreme outliers (e.g., values outside 1st and 99th percentiles)
q_low = last_60[['Open', 'High', 'Low', 'Close']].quantile(0.01)
q_high = last_60[['Open', 'High', 'Low', 'Close']].quantile(0.99)
mask = (last_60[['Open', 'High', 'Low', 'Close']] >= q_low) & (last_60[['Open', 'High', 'Low', 'Close']] <= q_high)
last_60 = last_60[mask.all(axis=1)]

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

# Optionally, set y-axis range to focus on typical price range
fig.update_layout(
    yaxis=dict(
        autorange=False,
        range=[
            last_60['Low'].min() * 0.98,
            last_60['High'].max() * 1.02
        ]
    )
)

# Get the last row
last_row = last_60.iloc[-1]

# Check for NaN values in the last row
nan_columns = last_row[last_row.isna()].index.tolist()

if nan_columns:
    print(f"NaN found in columns: {nan_columns}")
else:
    print("No NaN in the last row.")

# Suppose you have a scaler trained on 38 features
# X is your input, make sure it has shape (1, 38)
X = np.array(your_input).reshape(1, 38)
scaled_X = scaler.transform(X)

# ...existing code...