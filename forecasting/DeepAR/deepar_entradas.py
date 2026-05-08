import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sqlite3
import os
from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TIME_COL         = "date"
TARGET           = "Close"
FORECAST_HORIZON = 5       # semanas à frente
INPUT_SIZE       = 52      # ~1 ano de contexto semanal

# Níveis de confiança para o intervalo probabilístico
LEVELS = [80, 90]

# Threshold para decisão de compra/venda baseado na mediana prevista
THRESHOLD_COMPRA = 0.02    # +2%
THRESHOLD_VENDA  = -0.02   # -2%


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
# DeepAR: criar modelo
# ──────────────────────────────────────────────
def criar_modelo_deepar():
    """
    DeepAR com distribuição StudentT — mais robusta a outliers que Gaussian.

    Parâmetros chave:
      - lstm_n_layers: profundidade do LSTM (2 é padrão, aumentar = mais capacidade)
      - lstm_hidden_size: tamanho do estado oculto do LSTM
      - trajectory_samples: nº de trajetórias Monte Carlo na inferência
        (mais trajetórias = distribuição mais estável, mas mais lento)
      - loss=DistributionLoss('StudentT'): otimiza a log-likelihood da StudentT
      - valid_loss=MQLoss: valida com quantile loss (mais interpretável)
      - scaler_type='standard': normaliza a série — importante para LSTM
    """
    model = DeepAR(
        h=FORECAST_HORIZON,
        input_size=INPUT_SIZE,
        lstm_n_layers=2,
        lstm_hidden_size=128,
        lstm_dropout=0.1,
        trajectory_samples=100,
        loss=DistributionLoss(
            distribution="StudentT",
            level=LEVELS,
            return_params=False,
        ),
        valid_loss=MQLoss(level=LEVELS),
        max_steps=200,
        learning_rate=0.001,
        val_check_steps=20,
        early_stop_patience_steps=5,   # para após 5 validações sem melhora
        scaler_type="standard",        # normalização essencial para LSTM
        random_seed=42,
    )
    return model


def deepar_forecast(stock_df, id_ticker):
    """
    Recebe DataFrame com ['date', 'Close'] e retorna forecast probabilístico.
    Colunas de saída: ds, DeepAR-median, DeepAR-lo-80, DeepAR-hi-80,
                      DeepAR-lo-90, DeepAR-hi-90
    """
    df_nf = stock_df[["date", TARGET]].copy()
    df_nf = df_nf.rename(columns={"date": "ds", TARGET: "y"})
    df_nf["unique_id"] = id_ticker
    df_nf["ds"] = pd.to_datetime(df_nf["ds"])
    df_nf = df_nf.sort_values("ds").reset_index(drop=True)

    model = criar_modelo_deepar()
    nf = NeuralForecast(models=[model], freq="W")
    # val_size reserva os últimos FORECAST_HORIZON pontos para validação,
    # necessário para o early stopping funcionar
    nf.fit(df_nf, val_size=FORECAST_HORIZON)

    forecast = nf.predict()
    return forecast



class PredictRequest(BaseModel):
    tickers_analise: list[int]
    data: str | None = None


@app.post("/predict")
def main(req: PredictRequest):

    print("EXECUTANDO DEEPAR")

    engine   = getEngine()

    resultados = []

    data = req.data
    tickers = req.tickers_analise
    for id_ticker in tickers: 
        
        stock = getStockRange(id_ticker, engine, "2000-01-01", data)
        stock = resample_para_semanal(stock)

        if stock.empty or len(stock) < INPUT_SIZE + 10:
            print(f"{id_ticker}: Poucos dados, pulando...")
            continue

        stock = stock[stock["date"] <= pd.to_datetime(data)].copy()

        try:
            forecast = deepar_forecast(stock, id_ticker)
        except Exception as e:
            print(f"{id_ticker}: Erro no forecast — {e}")
            continue

        preco_atual = stock["Close"].iloc[-1]

        # DeepAR retorna mediana e intervalos
        mediana_prevista = forecast["DeepAR-median"].iloc[-1]

        variacao_prevista = (mediana_prevista - preco_atual) / preco_atual

        # Intervalo de confiança: verifica se o piso de 80% ainda é positivo
        lower_80 = forecast["DeepAR-lo-80"].iloc[-1]
        variacao_lower_80 = (lower_80 - preco_atual) / preco_atual

        # Decisão: mediana aponta alta E mesmo cenário pessimista (80%) é positivo
        if variacao_prevista > THRESHOLD_COMPRA and variacao_lower_80 > 0:
            sinal = 1
        elif variacao_prevista < THRESHOLD_VENDA:
            sinal = -1
        else:
            sinal = 0

        resultados.append({
            "id_ticker":         id_ticker,
            "mediana_prevista":  mediana_prevista,
            "variacao_prevista": variacao_prevista,
            "lower_80":          lower_80,
            "variacao_lower_80": variacao_lower_80,
            "sinal":             sinal
        })

    df_resultados = pd.DataFrame(resultados)
    return df_resultados.to_dict(orient="records")