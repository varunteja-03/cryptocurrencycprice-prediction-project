# cryptocurrencycprice-prediction-project
Project Overview:
This project is an AI-powered cryptocurrency market intelligence system that predicts future price trends using deep learning. It leverages historical crypto data and advanced feature engineering techniques to provide actionable insights for traders and analysts.

The system uses a trained neural network model to analyze patterns in cryptocurrency price movements and generate predictions with confidence levels.

Key Features:
Cryptocurrency Price Prediction
Deep learning model for time-series forecasting
Technical indicators like RSI (Relative Strength Index)
Multi-coin analysis support
Log return-based feature engineering
Interactive web interface using Streamlit
Real-time data fetching using Yahoo Finance

Tech Stack:
Python
Data Handling: Pandas, NumPy
Deep Learning: TensorFlow / Keras
Frontend UI: Streamlit
Data Source: yfinance
Model Persistence: Joblib, Pickle

Project Structure:
project123/
│
├── data/
│   ├── BTC-USD.csv
│   ├── crypto_3coins.csv
│   ├── crypto_multi_coin.csv
│   └── download_crypto_data.py
│
├── frontend/
│   └── app.py
│
├── model/
│   ├── saved_model.h5
│   ├── train_model.py
│   ├── predict.py
│   ├── scaler.pkl
│   ├── coins.pkl
│   ├── feature_names.pkl
│   ├── timestep.pkl
│   └── other model files
│
└── requirements.txt

How It Works:
1. Data Collection
Historical crypto data is collected using yfinance
Multiple coins can be analyzed together
2. Feature Engineering
Log returns calculation
RSI (Relative Strength Index)
Volatility features
3. Model Training
Deep learning model trained on time-series data
Uses sequences (timesteps = 60) for prediction
4. Prediction
Input data is scaled using saved scalers
Model predicts future price trends
Confidence threshold applied for reliability
5. Visualization
Streamlit dashboard displays predictions and insights

Installation & Setup:
Step 1: Clone Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Application
cd frontend
streamlit run app.py

Dataset:
Cryptocurrency historical price data
Includes multiple coins like Bitcoin and others
Stored locally in CSV format

Model Details:
Model Type: Deep Learning (LSTM-based likely)
Input: Time-series sequences (60 timesteps)
Output: Price trend prediction
Stored as: saved_model.h5

Use Cases:
Crypto traders for decision support
Financial analysts
AI/ML learning projects
Time-series forecasting research
