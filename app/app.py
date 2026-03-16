import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="BTC Trend Prediction", layout="wide")

st.title("🚀 BTC Daily Price Trend Prediction")
st.markdown("""
Predicting Bitcoin price movements using machine learning.
This interface allows you to visualize data and run model predictions.
""")

# Load data placeholder
# data = pd.read_csv('data/raw/btc_usd.csv')
# st.line_chart(data['Close'])

st.sidebar.header("Settings")
st.sidebar.info("Phase 6 implementation.")
