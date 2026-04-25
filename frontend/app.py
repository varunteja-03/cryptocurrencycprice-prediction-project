import streamlit as st


st.set_page_config(page_title="Crypto price prediction", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_all():
    model = load_model(os.path.join(BASE_DIR, "../model/saved_model.h5"), compile=False)
    scaler = joblib.load(os.path.join(BASE_DIR, "../model/scaler.pkl"))
    coins = joblib.load(os.path.join(BASE_DIR, "../model/coins.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "../model/feature_names.pkl"))
    return model, scaler, coins, features

model, scaler, coins, trained_features = load_all()

TIMESTEP = 60
CONF_THRESHOLD = 60   # slightly relaxed (better signals)


def build_features(df, coins_list):
    data = np.log(df / df.shift(1))

    for coin in coins_list:
        delta = df[coin].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss

        data[f'{coin}_RSI'] = 100 - (100 / (1 + rs))
        data[f'{coin}_VOL'] = data[coin].rolling(7).std()
        data[f'{coin}_MOM'] = df[coin].pct_change(3)

        data[f'{coin}_MA5'] = df[coin].rolling(5).mean() / df[coin]
        data[f'{coin}_MA10'] = df[coin].rolling(10).mean() / df[coin]

    return data.dropna()


st.title("Cryptocurrency Price Predictor")

choice = st.sidebar.selectbox("Select Coin", coins)
num_days = st.sidebar.slider("Days to Predict", 1, 30, 7)


ticker = yf.Ticker(choice)

hist_1d = ticker.history(period="1d", interval="1m")
if not hist_1d.empty:
    live_price = hist_1d["Close"].dropna().iloc[-1]
else:
    live_price = ticker.history(period="5d")["Close"].iloc[-1]

st.metric(f"Current {choice}", f"${live_price:,.2f}")


if st.button("Predict Future Prices"):

    with st.spinner("⏳ Predicting... please wait"):

        hist = yf.download(coins, period="150d")['Close'][coins]
        sim_data = hist.copy()

        preds = []
        signals = []
        confidences = []

        prev_pred = None

        for i in range(num_days):
            feats = build_features(sim_data, coins)
            feats = feats.reindex(columns=trained_features).dropna()

            if len(feats) < TIMESTEP:
                st.error("Not enough data to make predictions.")
                st.stop()

            scaled_in = scaler.transform(feats.tail(TIMESTEP))

            pred_reg, pred_dir = model.predict(
                scaled_in.reshape(1, TIMESTEP, -1),
                verbose=0
            )

            pred_log_return = pred_reg[0]

            # smoother prediction
            pred_log_return = np.clip(pred_log_return, -0.05, 0.05)

            if prev_pred is not None:
                pred_log_return = 0.75 * pred_log_return + 0.25 * prev_pred
            prev_pred = pred_log_return

            decay = 1 / (1 + 0.15 * i)
            pred_log_return *= decay

            last_log_price = np.log(sim_data.iloc[-1].values)
            future_prices = np.exp(last_log_price + pred_log_return)

            noise = np.random.normal(0, 0.0015, size=len(future_prices))
            future_prices = future_prices * (1 + noise)

            pred_price = future_prices[coins.index(choice)]
            preds.append(pred_price)

           
            coin_idx = coins.index(choice)
            prob_up = pred_dir[0][coin_idx]

            confidence = max(prob_up, 1 - prob_up) * 100

            if confidence < CONF_THRESHOLD:
                signal = "HOLD"
            elif prob_up > 0.52:
                signal = "BUY"
            elif prob_up < 0.48:
                signal = "SELL"
            else:
                signal = "HOLD"

            signals.append(signal)
            confidences.append(confidence)

            # update data
            new_row = pd.Series(
                {c: future_prices[coins.index(c)] for c in coins},
                name=hist.index[-1] + pd.Timedelta(days=i+1)
            )
            sim_data = pd.concat([sim_data, new_row.to_frame().T])

       
        future_dates = pd.date_range(
            hist.index[-1] + pd.Timedelta(days=1),
            periods=num_days
        )

        price_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": preds
        })

        st.subheader("Predicted Prices")
        st.dataframe(price_df.style.format({"Predicted Price": "${:,.2f}"}))

        st.subheader("Prediction Chart")
        st.line_chart(pd.Series(preds, index=future_dates))

        signal_df = pd.DataFrame({
            "Signal": signals,
            "Confidence %": confidences
        }, index=future_dates.strftime('%Y-%m-%d'))

        st.subheader(" Trading Signals")
        st.dataframe(signal_df.style.format({"Confidence %": "{:.2f}"}))

        final_signal = signals[0]
        final_conf = confidences[0]

        if final_signal == "BUY":
            st.success(f"BUY ({final_conf:.2f}%)")
        elif final_signal == "SELL":
            st.error(f"SELL ({final_conf:.2f}%)")
        else:
            st.warning(f"HOLD ({final_conf:.2f}%)")