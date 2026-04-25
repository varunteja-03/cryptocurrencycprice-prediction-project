import yfinance as yf
import pandas as pd

coins = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "BNB-USD",
    "XRP-USD",
    "ADA-USD"
]

data = yf.download(coins, start="2017-01-01", group_by="ticker")

data.to_csv("crypto_multi_coin.csv")

print("✅ Dataset downloaded successfully")