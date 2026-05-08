# ==============================
# IMPORTS
# ==============================
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from chronos import ChronosPipeline
import os
from sqlalchemy import create_engine
import sqlite3

# ==============================
# CONFIG
# ==============================
TIME_COL = "date"
TARGET = "Close"
FORECAST_HORIZON = 5

app = FastAPI()


# ==============================
# DATABASE
# ==============================
def get_connection():
    conn = sqlite3.connect("C:\\Projetos\\Github_Position\\main\\position_strategies\\db\\database.db")
    conn.row_factory = sqlite3.Row
    return conn

def getEngine():
    host = os.getenv("MYSQL_HOST", "host.docker.internal")
    return create_engine(f"mysql+mysqlconnector://root:1234@{host}:3306/position_invest")

def getEntradas():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            e.data_confirmacao,
            e.preco_entrada,
            o.id_ticker,
            o.ticker
        FROM entradas e
        INNER JOIN oportunidades o
        ON e.oportunidade_id = o.id
    """)
    return cur.fetchall()

def getStockRange(id_ticker, engine, dataInicio, dataFim):
    query = """
        SELECT date, Close, High, Low, Open, Volume
        FROM stock
        WHERE ticker_id = %(id)s
        AND date >= %(inicio)s AND date <= %(fim)s
        ORDER BY date ASC
    """
    return pd.read_sql(query, engine, params={
        "id": id_ticker,
        "inicio": dataInicio,
        "fim": dataFim
    })

# ==============================
# SAMPLE 20 POR MÊS
# ==============================
def preparar_entradas():
    entradas = getEntradas()
    df = pd.DataFrame([dict(x) for x in entradas])

    df["data_confirmacao"] = pd.to_datetime(df["data_confirmacao"])

    df["ano_mes"] = df["data_confirmacao"].dt.to_period("M")

    def sample_mes(g):
        return g if len(g) <= 20 else g.sample(20, random_state=42)

    df = df.groupby("ano_mes", group_keys=False).apply(sample_mes)

    return df.reset_index(drop=True)

# ==============================
# CHRONOS
# ==============================
pipeline_chronos = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.float32,
)

def chronos_forecast(model, df, horizon):
    context = torch.tensor(df[TARGET].values, dtype=torch.float32)

    forecast = model.predict(context, horizon)

    # quantis
    lower, median, upper = np.quantile(
        forecast[0].numpy(),
        [0.1, 0.5, 0.9],
        axis=0
    )

    return lower, median, upper



def resample_para_semanal(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df_semanal = df.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna().reset_index()
    return df_semanal


class PredictRequest(BaseModel):
    tickers_analise: list[int]
    data: str | None = None


@app.post("/predict")
def main(req: PredictRequest):
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    print("EXECUTANDO CHRONOS")
    
    engine = getEngine()

    entradas_list = []
    
    data = req.data
    tickers = req.tickers_analise
    for id_ticker in tickers: 
        
        # =========================
        # LOAD DADOS
        # =========================
        stock = getStockRange(
            id_ticker,
            engine,
            '2000-01-01',
            data
        )

        stock = resample_para_semanal(stock)
        

        if stock.empty or len(stock) < 50:
            print("Poucos dados, pulando...")
            continue

        stock[TIME_COL] = pd.to_datetime(stock[TIME_COL])
        data_entrada_dt = pd.to_datetime(data)
        
        stock = stock[stock[TIME_COL] <= data_entrada_dt].copy()
        
        if len(stock) < 30:
            print(len(stock))
            print("Poucos dados após filtro, pulando...")
            continue
        
        train = stock.copy()
        
        # FORECAST
        try:
            lower, median, upper = chronos_forecast(
                pipeline,
                train,
                FORECAST_HORIZON
            )
        except Exception as e:
            print(f"Erro no Chronos: {e}")
            continue
        
        # FUTURO
        future_dates = pd.bdate_range(
            start=data_entrada_dt + pd.Timedelta(days=1),
            periods=FORECAST_HORIZON
        )
        
        forecast_df = pd.DataFrame({
            TIME_COL: future_dates,
            "forecast_lower": lower,
            "forecast": median,
            "forecast_upper": upper
        })

        inicio = forecast_df["forecast"].iloc[0]
        fim = forecast_df["forecast"].iloc[-1]
        
        sinal = 0
        if fim > inicio: 
            sinal = 1

        entradas_list.append({
            "id_ticker":id_ticker,
            "sinal": sinal
        })  
        
    entradas_df = pd.DataFrame(entradas_list)
    return entradas_df.to_dict(orient="records")