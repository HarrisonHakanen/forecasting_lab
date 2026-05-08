import pandas as pd
import numpy as np
import timesfm
import warnings
from sqlalchemy import create_engine
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
import os

TIME_COL = "date"
FORECAST_HORIZON = 12
FREQ = "W"

app = FastAPI()

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
    warnings.filterwarnings("ignore")

    engine = getEngine()
    data = req.data
    tickers = req.tickers_analise

    resultados = []

    # Modelo carregado uma única vez fora do loop
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            context_len=512,
            horizon_len=FORECAST_HORIZON,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu",
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
        ),
    )

    
    for id_ticker in tickers:

        # Dados históricos até a data do sinal — sem look-ahead
        stock = getStockRange(
            id_ticker,
            engine,
            '2000-01-01',
            data,
        )

        stock = resample_para_semanal(stock)

        if stock.empty or len(stock) < 50:
            print(f"{id_ticker}: Poucos dados, pulando...")
            continue

        stock[TIME_COL] = pd.to_datetime(stock[TIME_COL])
        stock = stock[stock[TIME_COL] <= pd.to_datetime(data)].copy()
        stock = stock.dropna(subset=["Close"])

        if len(stock) < 30:
            print(f"{id_ticker}: Poucos dados após filtro ({len(stock)}), pulando...")
            continue
            

        stock["unique_id"] = id_ticker
        stock = stock.rename(columns={TIME_COL: "ds"})

        # --- Forecast 1: Close bruto (sinal de curto prazo) ---
        forecast_close = tfm.forecast_on_df(
            inputs=stock[["ds", "unique_id", "Close"]].copy(),
            freq=FREQ,
            value_name="Close",
            num_jobs=1,
        ).rename(columns={
            "timesfm-q-0.5": "forecast",
            "timesfm-q-0.1": "forecast_lower",
            "timesfm-q-0.9": "forecast_upper",
        })

        preco_atual     = stock["Close"].iloc[-1]
        preco_previsto  = forecast_close["forecast"].iloc[-1]

        # Variação prevista de cada série
        variacao_close  = (preco_previsto  - preco_atual)  / preco_atual
        

        # Decisão: exige concordância entre Close bruto e tendência suavizada
        if variacao_close > 0.02:
            sinal = 1   # ambos apontam alta com convicção
        elif variacao_close < -0.02:
            sinal = -1  # ambos apontam queda
        else:
            sinal = 0   # divergência ou sinal fraco — neutro

        resultados.append({
            "id_ticker":            id_ticker,
            "preco_previsto":       preco_previsto,
            "variacao_close":       variacao_close,
            "sinal":                sinal
        })

    df_resultados = pd.DataFrame(resultados)
    return df_resultados.to_dict(orient="records")