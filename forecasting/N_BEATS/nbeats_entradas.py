import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from sqlalchemy import create_engine
import sqlite3
import matplotlib.pyplot as plt
import os
from fastapi import FastAPI
from pydantic import BaseModel

# ==============================
# CONFIG
# ==============================
TIME_COL = "date"
TARGET = "mm20"
FORECAST_HORIZON = 10

INPUT_CHUNK = 30
OUTPUT_CHUNK = FORECAST_HORIZON
N_EPOCHS = 20  # não deixa alto senão vai ficar lento

app = FastAPI()

# ==============================
# FUNÇÃO NBEATS
# ==============================
def nbeats_forecast(series):

    ts = TimeSeries.from_series(series)

    model = NBEATSModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        n_epochs=N_EPOCHS,
        random_state=42        
    )

    model.fit(ts, verbose=False)

    pred = model.predict(OUTPUT_CHUNK)

    return pred.values().flatten()


def get_connection():
    conn = sqlite3.connect("C:\\Projetos\\Github_Position\\main\\position_strategies\\db\\database.db")
    conn.row_factory = sqlite3.Row
    return conn


def getEngine():
    host = os.getenv("MYSQL_HOST", "host.docker.internal")
    return create_engine(f"mysql+mysqlconnector://root:1234@{host}:3306/position_invest")
    

def getEmpresas():
    engine = getEngine()
    query_ticker = """
    SELECT * FROM ticker WHERE ticker.bolsa = 'B3' OR ticker.bolsa = 'Nasdaq' OR ticker.bolsa = 'LondonStockExchange' OR ticker.bolsa = 'Xetra' OR ticker.bolsa = 'Frankfurt'
    """

    df_ticker = pd.read_sql(query_ticker, engine)

    return df_ticker


def getEntradas():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            e.id,
            e.oportunidade_id,
            e.indicador,
            e.data_confirmacao,
            e.preco_entrada,
            e.ativo,
            o.id_ticker,
            o.ticker
        FROM entradas e
        INNER JOIN oportunidades o
        ON	e.oportunidade_id = o.id
    """)
    entradas = cur.fetchall()
    return entradas


def getStockRange(id_ticker, engine, dataInicio, dataFim, order=True):

    query_stock = f"""
        SELECT
            date,
            Open,
            High,
            Low,
            Close,
            Volume
        FROM stock
        WHERE ticker_id = %(id_ticker)s
        AND date >= %(dataInicio)s AND date <= %(dataFim)s
        ORDER BY date DESC
    """

    params = {"id_ticker": id_ticker,"dataInicio":dataInicio,"dataFim":dataFim}

    df = pd.read_sql(query_stock, engine, params=params)

    if order:
        df = df.iloc[::-1].reset_index(drop=True)

    return df


def preparar_entradas():
    entradas = getEntradas()
    df = pd.DataFrame([dict(x) for x in entradas])

    df["data_confirmacao"] = pd.to_datetime(df["data_confirmacao"])

    # remove duplicatas
    df = df.drop_duplicates(subset=["ticker", "data_confirmacao"])

    df["ano_mes"] = df["data_confirmacao"].dt.to_period("M")

    def sample_mes(g):
        return g if len(g) <= 20 else g.sample(20, random_state=42)

    df = df.groupby("ano_mes", group_keys=False).apply(sample_mes)

    return df.reset_index(drop=True)


def plot_previsao_vs_real(engine, df_resultados, n_amostras=5):
    
    amostras = df_resultados.sample(n=min(n_amostras, len(df_resultados)), random_state=42)

    for _, row in amostras.iterrows():
        
        ticker = row["ticker"]
        data = pd.to_datetime(row["data"])
        id_ticker = row["id_ticker"]

        # histórico
        hist = getStockRange(
            id_ticker,
            engine,
            data - pd.Timedelta(days=120),
            data
        )

        # futuro real
        futuro = getStockRange(
            id_ticker,
            engine,
            data,
            data + pd.Timedelta(days=90)
        )

        if len(hist) < 30 or len(futuro) < FORECAST_HORIZON:
            continue

        hist[TIME_COL] = pd.to_datetime(hist[TIME_COL])
        futuro[TIME_COL] = pd.to_datetime(futuro[TIME_COL])

        hist = hist.sort_values(TIME_COL)
        futuro = futuro.sort_values(TIME_COL)

        # série MA20
        hist["mm20"] = hist["Close"].rolling(20).mean()
        futuro["mm20"] = futuro["Close"].rolling(20).mean()

        hist = hist.dropna()
        futuro = futuro.dropna()

        # reconstruir previsão com NBEATS
        serie = hist["mm20"]
        ts = TimeSeries.from_series(serie)

        model = NBEATSModel(
            input_chunk_length=30,
            output_chunk_length=FORECAST_HORIZON,
            n_epochs=20,
            random_state=42,
        )

        model.fit(ts, verbose=False)
        pred = model.predict(FORECAST_HORIZON)

        pred_df = pd.DataFrame({
            TIME_COL: pred.time_index,
            "forecast": pred.values().flatten()
        })
        pred_df.columns = [TIME_COL, "forecast"]

        # ======================
        # PLOT
        # ======================
        plt.figure(figsize=(12,5))

        plt.plot(hist[TIME_COL], hist["mm20"], label="Histórico", color="blue")
        plt.plot(futuro[TIME_COL].iloc[:FORECAST_HORIZON], 
                 futuro["mm20"].iloc[:FORECAST_HORIZON],
                 label="Real Futuro", linestyle="dashed", color="green")

        plt.plot(pred_df[TIME_COL], pred_df["forecast"], 
                 label="Forecast NBEATS", color="red")

        plt.axvline(x=data, color="black", linestyle=":", label="Entrada")

        plt.title(f"{ticker} - Previsão vs Real")
        plt.legend()
        plt.grid()

        plt.show()


def resample_para_semanal(df):
    """
    Converte DataFrame diário (OHLCV) para semanal.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df_semanal = df.resample("W").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum"
    }).dropna().reset_index()

    return df_semanal



def carregar_entradas():
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT
                e.id               AS entrada_id,
                e.oportunidade_id,
                e.indicador,
                e.data_confirmacao,
                e.preco_entrada,
                o.id_ticker,
                o.ticker
            FROM entradas e
            INNER JOIN oportunidades o ON e.oportunidade_id = o.id
            WHERE e.preco_entrada IS NOT NULL
              AND e.preco_entrada > 0
            ORDER BY e.data_confirmacao, e.id
        """).fetchall()


    finally:
        conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    df["data_confirmacao"] = pd.to_datetime(df["data_confirmacao"])
    return df
    

def remover_duplicados(df):
    df = df.sort_values(["ticker", "data_confirmacao"])
    df["diff_dias"] = df.groupby("ticker")["data_confirmacao"].diff().dt.days
    df_filtrado = df[(df["diff_dias"].isna()) | (df["diff_dias"] > 1)]

    return df_filtrado.drop(columns="diff_dias")



class PredictRequest(BaseModel):
    tickers_analise: list[int]
    data: str | None = None


@app.post("/predict")
def main(req: PredictRequest):

    print("EXECUTANDO NBEATS")

    engine = getEngine()

    resultados = []

    data = req.data
    tickers = req.tickers_analise
    for id_ticker in tickers: 

        stock = getStockRange(id_ticker, engine, "2010-01-01", data)
        stock = resample_para_semanal(stock)
    

        if len(stock) < 60:
            continue
        
        stock[TIME_COL] = pd.to_datetime(stock[TIME_COL])
        stock = stock.sort_values(TIME_COL)

        # ======================
        # RET
        # ======================
        stock["ret"] = stock["Close"].pct_change()
        stock = stock.dropna(subset=["ret"])
        
        serie = stock["Close"]

        # ======================
        # NBEATS
        # ======================
        print("Realizando previsões")
        preco_atual = stock[TARGET].iloc[-1]
        preco_previsto = nbeats_forecast(serie)[-1]
        

        variacao_close  = (preco_previsto  - preco_atual)  / preco_atual
        
        if variacao_close > 0.02:
            sinal = 1
        elif variacao_close < -0.02:
            sinal = -1
        else:
            sinal = 0
        
        resultados.append({     
            "id_ticker": id_ticker,
            "sinal": sinal,
            "preco_previsto": preco_previsto
        })


    df = pd.DataFrame(resultados)
    return df.to_dict(orient="records")