import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
import os

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import MQLoss

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FORECAST_HORIZON = 5       # semanas à frente
INPUT_SIZE       = 52      # ~1 ano de contexto semanal

# Threshold para decisão (sobre a mediana prevista)
THRESHOLD_COMPRA = 0.02    # +2%
THRESHOLD_VENDA  = -0.02   # -2%

# Covariáveis futuras conhecidas (calendário) — únicas para o TFT
# São conhecidas no futuro pois são datas do calendário
FUTR_EXOG = ["month", "week_of_year", "day_of_week"]

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


def adicionar_features_calendario(df, ticker, horizonte=FORECAST_HORIZON):
    """
    Adiciona features de calendário (mês, semana, dia da semana) ao DataFrame
    histórico e cria as linhas futuras necessárias para o TFT prever.

    O TFT precisa das covariáveis futuras para todo o horizonte de previsão —
    por isso criamos as próximas `horizonte` semanas explicitamente.
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["date"])   # remove coluna Timestamp — neuralforecast só aceita "ds"
    df["unique_id"] = ticker
    df = df.rename(columns={"Close": "y"})

    # Adiciona features de calendário no histórico
    df["month"]        = df["ds"].dt.month
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["day_of_week"]  = df["ds"].dt.dayofweek

    # Cria as linhas futuras (horizonte) com as covariáveis de calendário
    ultima_data  = df["ds"].iloc[-1]
    datas_futuras = pd.date_range(
        start=ultima_data + pd.Timedelta(weeks=1),
        periods=horizonte,
        freq="W"
    )

    df_futuro = pd.DataFrame({
        "ds":           datas_futuras,
        "unique_id":    ticker,
        "y":            np.nan,               # alvo desconhecido no futuro
        "month":        datas_futuras.month,
        "week_of_year": datas_futuras.isocalendar().week.astype(int).values,
        "day_of_week":  datas_futuras.dayofweek,
    })

    df_completo = pd.concat([df, df_futuro], ignore_index=True)
    return df_completo


# ──────────────────────────────────────────────
# TFT: criar modelo
# ──────────────────────────────────────────────
def criar_modelo_tft():
    """
    TFT com covariáveis de calendário futuras.

    Parâmetros chave:
      - hidden_size: dimensão dos vetores internos (32-512)
      - n_head: cabeças de atenção multi-head (deve dividir hidden_size)
      - futr_exog_list: features conhecidas no futuro (mês, semana, dia)
      - loss=MQLoss: produz intervalos de confiança (80% e 90%)
      - scaler_type='standard': normalização por série — importante para TFT
    """
    model = TFT(
        h=FORECAST_HORIZON,
        input_size=INPUT_SIZE,
        hidden_size=64,
        n_head=4,
        futr_exog_list=FUTR_EXOG,
        loss=MQLoss(level=[80, 90]),
        max_steps=200,
        learning_rate=0.001,
        val_check_steps=20,
        early_stop_patience_steps=-1,
        scaler_type="standard",
        random_seed=42,
    )
    return model


def tft_forecast(stock_df, ticker):
    """
    Recebe DataFrame com ['date', 'Close'] e retorna forecast probabilístico.

    Colunas de saída:
        TFT-median  → mediana (previsão central)
        TFT-lo-80   → limite inferior 80% (pessimista)
        TFT-hi-80   → limite superior 80% (otimista)
        TFT-lo-90   → limite inferior 90%
        TFT-hi-90   → limite superior 90%
    """
    df_completo = adicionar_features_calendario(stock_df, ticker)

    # Histórico (y preenchido) — dados de treino
    df_hist  = df_completo[df_completo["y"].notna()].copy()

    # Futuro (y=NaN) — covariáveis para as semanas previstas
    df_futur = df_completo[df_completo["y"].isna()].copy()

    model = criar_modelo_tft()
    nf = NeuralForecast(models=[model], freq="W")
    nf.fit(df_hist)

    # futr_df fornece as covariáveis futuras para o horizonte
    forecast = nf.predict(futr_df=df_futur)
    return forecast



class PredictRequest(BaseModel):
    tickers_analise: list[int]
    data: str | None = None


@app.post("/predict")
def main(req: PredictRequest):

    print("EXECUTANDO TFT")

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
            forecast = tft_forecast(stock, id_ticker)
        except Exception as e:
            print(f"{id_ticker}: Erro no forecast — {e}")
            continue

        preco_atual      = stock["Close"].iloc[-1]
        mediana_prevista = forecast["TFT-median"].iloc[-1]
        lower_10         = forecast["TFT-lo-80"].iloc[-1]

        variacao_prevista = (mediana_prevista - preco_atual) / preco_atual
        variacao_lower    = (lower_10 - preco_atual) / preco_atual

        # Compra: mediana positiva E cenário pessimista (q10) também positivo
        if variacao_prevista > THRESHOLD_COMPRA and variacao_lower > 0:
            sinal = 1
        elif variacao_prevista < THRESHOLD_VENDA:
            sinal = -1
        else:
            sinal = 0

        resultados.append({
            "mediana_prevista":  mediana_prevista,
            "variacao_prevista": variacao_prevista,
            "lower_q10":         lower_10,
            "variacao_lower":    variacao_lower,
            "sinal":             sinal
        })

    df_resultados = pd.DataFrame(resultados)
    return df_resultados.to_dict(orient="records")