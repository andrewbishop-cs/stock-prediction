import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime

# Cache data loading for performance
@st.cache_data
def load_features(path='data/processed/features_final.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    return df

@st.cache_resource
def load_model(path='models/gbm_tuned.pkl'):
    return joblib.load(path)

# Load data and model
features = load_features()
model = load_model()

def main():
    st.title('📈 Stock Movement Predictor')
    
    # Sidebar - controls
    st.sidebar.header('Controls')
    tickers = sorted(features['ticker'].unique())
    ticker = st.sidebar.selectbox('Select Ticker', tickers)

    min_date = features['date'].min().date()
    max_date = features['date'].max().date()
    start_date, end_date = st.sidebar.date_input(
        'Date range', [min_date, max_date], min_value=min_date, max_value=max_date
    )

    threshold = st.sidebar.slider('Probability Threshold', 0.0, 1.0, 0.46, 0.01)

    # Filter features
    mask = (
        (features['ticker'] == ticker) & 
        (features['date'] >= pd.to_datetime(start_date)) &
        (features['date'] <= pd.to_datetime(end_date))
    )
    df_sel = features[mask].copy()

    if df_sel.empty:
        st.warning('No data for selected ticker and date range.')
        return

    # Predict probabilities
    X = df_sel.drop(['date','ticker','target','text_raw'], axis=1)
    df_sel['prob_up'] = model.predict_proba(X)[:, 1]
    df_sel['pred_up'] = (df_sel['prob_up'] >= threshold).astype(int)

    # Price chart
    st.subheader(f'{ticker} Close Price')
    price_data = yf.download(ticker, start=start_date, end=end_date)
    if not price_data.empty:
        st.line_chart(price_data['Close'])

    # Prediction probability chart
    st.subheader('Predicted Probability of Up Move')
    prob_chart = df_sel.set_index('date')['prob_up']
    st.line_chart(prob_chart)

    # Predicted class chart
    st.subheader(f'Predicted Class (Threshold = {threshold:.2f})')
    st.bar_chart(df_sel.set_index('date')['pred_up'])

    # Show raw table
    if st.sidebar.checkbox('Show raw data'):
        st.subheader('Raw Features & Predictions')
        st.write(df_sel)

if __name__ == '__main__':
    main()
