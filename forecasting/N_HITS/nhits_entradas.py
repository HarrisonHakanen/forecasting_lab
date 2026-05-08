import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
import os

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TIME_COL         = "date"
TARGET           = "Close"
FORECAST_HORIZON = 5      # semanas à frente
INPUT_CHUNK      = 52     # ~1 ano de dados semanais como contexto

# Threshold para decisão de compra/venda (variação % prevista)
THRESHOLD_COMPRA = 0.02   # +2%
THRESHOLD_VENDA  = -0.02  # -2%

app = FastAPI()

# ──────────────────────────────────────────────
# CONEXÃO
# ──────────────────────────────────────────────
def get_connection():
    conn = sqlite3.connect(
        "C:\\Projetos\\Github_Position\\main\\position_strategies\\db\\database.db"
    )
    conn.row_factory = sqlite3.Row
    return conn


def getEngine():
    host = os.getenv("MYSQL_HOST", "host.docker.internal")
    return create_engine(f"mysql+mysqlconnector://root:1234@{host}:3306/position_invest")


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


def getStockRange(id_ticker, engine, dataInicio, dataFim):
    query = """
        SELECT date, Open, High, Low, Close, Volume
        FROM stock
        WHERE ticker_id = %(id_ticker)s
          AND date >= %(dataInicio)s AND date <= %(dataFim)s
        ORDER BY date ASC
    """
    params = {"id_ticker": id_ticker, "dataInicio": dataInicio, "dataFim": dataFim}
    return pd.read_sql(query, engine, params=params)


def resample_para_semanal(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df_semanal = df.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna().reset_index()
    return df_semanal

# ──────────────────────────────────────────────
# N-HiTS: criar e treinar o modelo
# ──────────────────────────────────────────────
def criar_modelo_nhits():
    """
    Cria o modelo N-HiTS com 3 stacks (padrão do paper).

    Parâmetros chave:
      - n_stacks=3: 3 níveis hierárquicos de frequência
      - n_blocks=[1,1,1]: 1 bloco por stack (aumentar para mais capacidade)
      - n_pool_kernel_size=[8,4,1]: pooling agressivo→suave (longo→curto prazo)
      - n_freq_downsample=[2,1,1]: interpolação hierárquica
      - max_steps: iterações de treinamento (aumentar para mais precisão, mas mais lento)
    """
    model = NHITS(
        h=FORECAST_HORIZON,
        input_size=INPUT_CHUNK,
        loss=MAE(),
        
        n_blocks=[1, 1, 1],
        mlp_units=[[256, 256], [256, 256], [256, 256]],
        n_pool_kernel_size=[8, 4, 1],   # multi-rate sampling
        n_freq_downsample=[2, 1, 1],    # hierarchical interpolation
        dropout_prob_theta=0.0,
        max_steps=100,
        val_check_steps=10,
        random_seed=42,
        # Para previsão probabilística, descomente:
        # loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
    )
    return model


def nhits_forecast(stock_df, ticker):
    """
    Recebe um DataFrame com colunas ['date', 'Close'] e retorna o forecast.
    A neuralforecast espera colunas: ds (data), y (target), unique_id.
    """
    df_nf = stock_df[["date", TARGET]].copy()
    df_nf = df_nf.rename(columns={"date": "ds", TARGET: "y"})
    df_nf["unique_id"] = ticker
    df_nf["ds"] = pd.to_datetime(df_nf["ds"])
    df_nf = df_nf.sort_values("ds").reset_index(drop=True)

    model = criar_modelo_nhits()
    nf = NeuralForecast(models=[model], freq="W")
    nf.fit(df_nf)

    forecast = nf.predict()
    # forecast tem colunas: unique_id, ds, NHITS
    return forecast


class PredictRequest(BaseModel):
    tickers_analise: list[int]
    data: str | None = None


@app.post("/predict")
def main(req: PredictRequest):

    print("EXECUTANDO NHITS")

    engine   = getEngine()
    resultados = []

    data = req.data
    tickers = req.tickers_analise
    for id_ticker in tickers:
        
        stock = getStockRange(id_ticker, engine, "2000-01-01", data)
        stock = resample_para_semanal(stock)

        if stock.empty or len(stock) < INPUT_CHUNK + 10:
            print(f"{id_ticker}: Poucos dados, pulando...")
            continue

        stock = stock[stock["date"] <= pd.to_datetime(data)].copy()

        try:
            forecast = nhits_forecast(stock, id_ticker)
        except Exception as e:
            print(f"{id_ticker}: Erro no forecast — {e}")
            continue

        preco_atual    = stock["Close"].iloc[-1]
        preco_previsto = forecast["NHITS"].iloc[-1]

        variacao_prevista = (preco_previsto - preco_atual) / preco_atual

        if variacao_prevista > THRESHOLD_COMPRA:
            sinal = 1
        elif variacao_prevista < THRESHOLD_VENDA:
            sinal = -1
        else:
            sinal = 0

        resultados.append({
            "ticker":           id_ticker,
            "preco_previsto":   preco_previsto,
            "variacao_prevista": variacao_prevista,
            "sinal":            sinal
        })

    df_resultados = pd.DataFrame(resultados)
    return df_resultados.to_dict(orient="records")