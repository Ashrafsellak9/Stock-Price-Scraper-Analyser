"""
Analyse de cours boursiers : récupération des données, indicateurs et graphiques.
Projet perso orienté quant / analyse de marché.

Pour installer : pip install yfinance pandas matplotlib plotly
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# Récupération des cours
def fetch_data(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """
    Télécharge l'historique OHLCV (ouverture, plus haut, plus bas, clôture, volume)
    pour chaque symbole de la liste.
    period peut être "1y", "2y", "5y", "max", etc.
    """
    data = {}
    for ticker in tickers:
        print(f"Téléchargement : {ticker}")
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            print(f"Aucune donnée pour {ticker}")
            continue
        df.index = pd.to_datetime(df.index)
        data[ticker] = df
        print(f"{len(df)} jours | {df.index[0].date()} → {df.index[-1].date()}")
    return data


# Nettoyage des données
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare le tableau pour l'analyse :
    on enlève les doublons, on comble les trous (week-ends, jours fériés),
    et on retire les lignes où les prix ne collent pas (ex. plus haut < plus bas).
    """
    # Une seule ligne par date
    df = df[~df.index.duplicated(keep="first")]

    # Les trous sont souvent des jours sans cotation : on propage la dernière valeur connue
    df = df.ffill().bfill()

    # On garde seulement les jours où les prix ont du sens
    mask = (
        (df["High"] >= df["Low"]) &
        (df["High"] >= df["Open"]) &
        (df["High"] >= df["Close"]) &
        (df["Low"]  <= df["Open"]) &
        (df["Low"]  <= df["Close"]) &
        (df["Close"] > 0) &
        (df["Volume"] >= 0)
    )
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"{n_dropped} ligne(s) incohérente(s) supprimée(s)")
    return df[mask]


