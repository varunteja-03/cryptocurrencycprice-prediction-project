import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')


def build_features(df, coins):
    data = np.log(df / df.shift(1))
    for coin in coins:
        delta = df[coin].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

        data[f'{coin}_RSI'] = 100 - (100 / (1 + (gain/loss)))
        data[f'{coin}_VOL'] = data[coin].rolling(7).std()
        data[f'{coin}_MOM'] = df[coin].pct_change(3)

        data[f'{coin}_MA5'] = df[coin].rolling(5).mean() / df[coin]
        data[f'{coin}_MA10'] = df[coin].rolling(10).mean() / df[coin]

    return data.dropna()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "crypto_multi_coin.csv")

df_raw = pd.read_csv(DATA_PATH, header=[0, 1], index_col=0)
close_prices = df_raw.xs("Close", axis=1, level=1).dropna()
coins = close_prices.columns.tolist()

full_df = build_features(close_prices, coins)
feature_names = full_df.columns.tolist()

split_idx = int(0.8 * len(full_df))

scaler = RobustScaler()
scaler.fit(full_df.iloc[:split_idx])

scaled_data = scaler.transform(full_df)
scaled_data = np.clip(scaled_data, -3, 3)

TIMESTEP = 60

X, y_reg, y_dir = [], [], []

for i in range(TIMESTEP, len(scaled_data) - 1):
    X.append(scaled_data[i-TIMESTEP:i])

    future_return = np.log(
        (0.6 * close_prices.iloc[i+1] +
         0.3 * close_prices.iloc[i] +
         0.1 * close_prices.iloc[i-1]) / close_prices.iloc[i]
    ).values

    y_reg.append(future_return)

   
    threshold = 0.0035
    direction = np.where(future_return > threshold, 1,
                 np.where(future_return < -threshold, 0, -1))

    y_dir.append(direction)

X = np.array(X).astype('float32')
y_reg = np.array(y_reg).astype('float32')
y_dir = np.array(y_dir).astype('float32')

mask = (y_dir != -1).all(axis=1)
X = X[mask]
y_reg = y_reg[mask]
y_dir = y_dir[mask]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]
y_dir_train, y_dir_test = y_dir[:split], y_dir[split:]

early_stop = EarlyStopping(patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)


inputs = Input(shape=(TIMESTEP, X.shape[2]))

x = GRU(192, return_sequences=True)(inputs)
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x = GRU(96)(x)

x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

gru_reg = Dense(len(coins), name="regression", dtype='float32')(x)
gru_dir = Dense(len(coins), activation='sigmoid', name="direction", dtype='float32')(x)

gru_model = Model(inputs=inputs, outputs=[gru_reg, gru_dir])

gru_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
    loss={
        "regression": tf.keras.losses.Huber(delta=0.025),
        "direction": "binary_crossentropy"
    },
    loss_weights={
        "regression": 1.1,
        "direction": 1.9   # 🔥 Strong focus on direction
    }
)

gru_model.fit(
    X_train,
    {"regression": y_reg_train, "direction": y_dir_train},
    validation_data=(X_test, {"regression": y_reg_test, "direction": y_dir_test}),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

gru_pred_reg, gru_pred_dir = gru_model.predict(X_test)


gru_pred_reg = np.clip(gru_pred_reg, -0.06, 0.06)
gru_pred_reg = 0.92 * gru_pred_reg + 0.08 * np.roll(gru_pred_reg, 1, axis=0)

gru_pred_dir = 0.98 * gru_pred_dir + 0.02 * np.roll(gru_pred_dir, 1, axis=0)


gru_rmse = np.sqrt(mean_squared_error(y_reg_test, gru_pred_reg))
gru_mae = mean_absolute_error(y_reg_test, gru_pred_reg)

gru_dir_acc = accuracy_score(
    y_dir_test.flatten(),
    (gru_pred_dir > 0.46).astype(int).flatten()
)

print(f"GRU MAE: {gru_mae:.6f}")
print(f"GRU RMSE: {gru_rmse:.6f}")
print(f"GRU Direction Accuracy: {gru_dir_acc:.4f}")


lstm_inputs = Input(shape=(TIMESTEP, X.shape[2]))

lx = LSTM(96, return_sequences=True)(lstm_inputs)
lx = BatchNormalization()(lx)
lx = Dropout(0.3)(lx)

lx = LSTM(48)(lx)
lx = Dense(64, activation='relu')(lx)

lstm_reg = Dense(len(coins), name="regression", dtype='float32')(lx)
lstm_dir = Dense(len(coins), activation='sigmoid', name="direction", dtype='float32')(lx)

lstm_model = Model(inputs=lstm_inputs, outputs=[lstm_reg, lstm_dir])

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss={
        "regression": tf.keras.losses.Huber(delta=0.05),
        "direction": "binary_crossentropy"
    },
    loss_weights={"regression": 1.2, "direction": 0.8}
)

lstm_model.fit(
    X_train,
    {"regression": y_reg_train, "direction": y_dir_train},
    validation_data=(X_test, {"regression": y_reg_test, "direction": y_dir_test}),
    epochs=120,
    batch_size=64,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

lstm_pred_reg, lstm_pred_dir = lstm_model.predict(X_test)

lstm_mae = mean_absolute_error(y_reg_test, lstm_pred_reg)
lstm_rmse = np.sqrt(mean_squared_error(y_reg_test, lstm_pred_reg))
lstm_dir_acc = accuracy_score(
    y_dir_test.flatten(),
    (lstm_pred_dir > 0.5).astype(int).flatten()
)

print(f"LSTM MAE: {lstm_mae:.6f}")
print(f"LSTM RMSE: {lstm_rmse:.6f}")
print(f"LSTM Direction Accuracy: {lstm_dir_acc:.4f}")

print("\n Model Comparison:")
print(f"GRU MAE: {gru_mae:.6f} | LSTM MAE: {lstm_mae:.6f}")
print(f"GRU RMSE: {gru_rmse:.6f} | LSTM RMSE: {lstm_rmse:.6f}")
print(f"GRU Direction: {gru_dir_acc:.4f} | LSTM Direction: {lstm_dir_acc:.4f}")

print("GRU selected (better performance)")

gru_model.save(os.path.join(BASE_DIR, "saved_model.h5"))

joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
joblib.dump(coins, os.path.join(BASE_DIR, "coins.pkl"))
joblib.dump(feature_names, os.path.join(BASE_DIR, "feature_names.pkl"))

print("Training Complete.")