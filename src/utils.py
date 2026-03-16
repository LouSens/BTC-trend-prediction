import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_btc_trend(df, title="BTC Price Trend"):
    """
    Custom plotter for BTC price trends.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='BTC Close Price')
    plt.title(title)
    plt.legend()
    plt.show()
