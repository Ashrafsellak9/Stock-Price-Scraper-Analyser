import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Fonction pour télécharger les données des tickers
def fetch_data(ticker: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in tickers:
        print("Téléchargement des données pour le ticker: ", ticker)
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            print(f"Aucune donnée trouvée pour le ticker: {ticker}")
        df.index = pd.to_datetime(df.index)
        data[ticker] = df
        print(f"{len(df)} jours | {df.index[0].date()} → {df.index[-1].date()}")
    return data

