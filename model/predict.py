import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model

# load model & scaler
model = load_model("saved_model.h5")
scaler = joblib.load("scaler.pkl")

# load latest data
df = yf.download("BTC-USD", period="60d")

data = df[['Close']].values

scaled_data = scaler.transform(data)

# last 30 days
x_input = scaled_data[-30:]
x_input = np.reshape(x_input, (1, 30, 1))

prediction = model.predict(x_input)

prediction = scaler.inverse_transform(prediction)

print("📈 Next day BTC price:", prediction[0][0])