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


# ─────────────────────────────────────────────
# Récupération des cours
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Nettoyage des données
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Indicateurs techniques
# ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le tableau avec rendements, volatilité, moyennes mobiles,
    MACD, RSI, bandes de Bollinger et drawdown.
    """
    df = df.copy()
    close = df["Close"]

    # Variation du cours d'un jour à l'autre
    df["log_return"]    = np.log(close / close.shift(1))
    df["simple_return"] = close.pct_change()

    # Volatilité sur 20 et 60 séances, ramenée à l'échelle annuelle (252 jours de bourse)
    df["vol_20d"]  = df["log_return"].rolling(20).std()  * np.sqrt(252)
    df["vol_60d"]  = df["log_return"].rolling(60).std()  * np.sqrt(252)

    # Moyennes mobiles simples et exponentielles
    df["sma_20"]  = close.rolling(20).mean()
    df["sma_50"]  = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()
    df["ema_12"]  = close.ewm(span=12, adjust=False).mean()
    df["ema_26"]  = close.ewm(span=26, adjust=False).mean()

    # MACD : écart entre deux EMA, avec sa ligne de signal et l'histogramme
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # RSI sur 14 jours (force relative, entre 0 et 100)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bandes de Bollinger : moyenne sur 20 jours ± 2 écarts-types
    df["bb_mid"]   = df["sma_20"]
    df["bb_upper"] = df["sma_20"] + 2 * close.rolling(20).std()
    df["bb_lower"] = df["sma_20"] - 2 * close.rolling(20).std()

    # Drawdown : à quel point le cours est en dessous de son plus haut historique
    roll_max       = close.cummax()
    df["drawdown"] = (close - roll_max) / roll_max

    return df


# ─────────────────────────────────────────────
# Bilan de performance
# ─────────────────────────────────────────────

def performance_metrics(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Résume la performance du titre : rendement, risque (volatilité, VaR),
    ratios Sharpe / Sortino et perte maximale depuis un sommet.
    """
    ret = df["log_return"].dropna()
    price = df["Close"].dropna()

    trading_days = 252
    ann_return   = ret.mean() * trading_days
    ann_vol      = ret.std()  * np.sqrt(trading_days)
    sharpe       = ann_return / ann_vol if ann_vol != 0 else np.nan

    # Sortino : comme le Sharpe, mais on ne pénalise que la volatilité des baisses
    downside     = ret[ret < 0].std() * np.sqrt(trading_days)
    sortino      = ann_return / downside if downside != 0 else np.nan

    roll_max     = price.cummax()
    drawdowns    = (price - roll_max) / roll_max
    max_drawdown = drawdowns.min()

    total_return = (price.iloc[-1] / price.iloc[0]) - 1
    var_95       = ret.quantile(0.05)          # perte journalière dépassée 5 % du temps
    cvar_95      = ret[ret <= var_95].mean()   # perte moyenne les jours les plus mauvais

    metrics = {
        "ticker"       : ticker,
        "période"      : f"{price.index[0].date()} → {price.index[-1].date()}",
        "nb_jours"     : len(price),
        "rendement_total" : f"{total_return:.1%}",
        "rendement_ann"   : f"{ann_return:.1%}",
        "volatilité_ann"  : f"{ann_vol:.1%}",
        "sharpe_ratio"    : f"{sharpe:.2f}",
        "sortino_ratio"   : f"{sortino:.2f}",
        "max_drawdown"    : f"{max_drawdown:.1%}",
        "VaR_95_1j"       : f"{var_95:.2%}",
        "CVaR_95_1j"      : f"{cvar_95:.2%}",
    }
    return metrics


# ─────────────────────────────────────────────
# Graphiques interactifs (Plotly)
# ─────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, ticker: str):
    """
    Construit une page HTML avec chandeliers, volume, MACD et RSI.
    Les barres vertes / rouges suivent le sens du cours du jour.
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.5, 0.15, 0.2, 0.15],
        subplot_titles=[
            f"{ticker} — Prix & Bollinger Bands",
            "Volume",
            "MACD",
            "RSI (14j)"
        ]
    )

    # Graphique principal : chandeliers + bandes et moyennes mobiles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    for col, color, name in [
        ("bb_upper", "rgba(100,120,255,0.3)", "BB Upper"),
        ("bb_lower", "rgba(100,120,255,0.3)", "BB Lower"),
        ("sma_50",   "rgba(255,180,0,0.8)",   "SMA 50"),
        ("sma_200",  "rgba(255,100,100,0.8)", "SMA 200"),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=name,
            line=dict(color=color, width=1), mode="lines"
        ), row=1, col=1)

    # Volume coloré selon que le cours a monté ou baissé ce jour-là
    colors = ["#26a69a" if r >= 0 else "#ef5350" for r in df["simple_return"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, showlegend=False
    ), row=2, col=1)

    # MACD : courbe, signal et histogramme
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],        name="MACD",   line=dict(color="#2196F3", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="#FF9800", width=1.2)), row=3, col=1)
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Histogramme", marker_color=hist_colors, showlegend=False), row=3, col=1)

    # RSI avec repères 30 (survente) et 70 (surachat)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI", line=dict(color="#9C27B0", width=1.2)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.5, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=4, col=1)

    fig.update_layout(
        title=f"Dashboard Quant — {ticker}",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    fig.write_html(f"{ticker}_dashboard.html")
    print(f"Dashboard sauvegardé : {ticker}_dashboard.html")
    fig.show()


# ─────────────────────────────────────────────
# Point d'entrée du script
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "NVDA"]
    PERIOD  = "2y"

    raw_data = fetch_data(TICKERS, period=PERIOD)

    all_metrics = []

    for ticker, df_raw in raw_data.items():
        print(f"\n{'─'*50}")
        print(f"  Traitement : {ticker}")
        print(f"{'─'*50}")

        df_clean = clean_data(df_raw)
        df = compute_indicators(df_clean)

        df.to_csv(f"{ticker}_data.csv")
        print(f"  CSV sauvegardé : {ticker}_data.csv")

        metrics = performance_metrics(df, ticker)
        all_metrics.append(metrics)
        print("\n  Métriques de performance :")
        for k, v in metrics.items():
            print(f"     {k:<20} {v}")

        plot_dashboard(df, ticker)

    # Tableau récap de tous les titres traités
    print(f"\n{'═'*50}")
    print("  RÉCAPITULATIF GLOBAL")
    print(f"{'═'*50}")
    df_recap = pd.DataFrame(all_metrics).set_index("ticker")
    print(df_recap.to_string())
