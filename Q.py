"""
Teste de matriz Q (Black-Litterman) para um único ativo (ABEV3),
usando previsão de 7h via modelo Lag-Llama já treinado e validação walk-forward.

Fluxo:
1) Lê série horária do ativo em tickers_data.
2) Calcula log_ret_vwap (target do modelo).
3) Para cada dia útil no período (ex.: 202506-202512):
   - monta contexto com histórico até o início do dia;
   - prevê 7 passos (7h) com Lag-Llama;
   - calcula retorno previsto do dia (soma dos 7 log-retornos).
4) Agrega sentimento do mesmo dia no JSON de notícias.
5) Combina previsão + sentimento para gerar Q diário.

Saídas:
- q_ab_ev3_walkforward_long.csv: componentes por dia.
- q_ab_ev3_walkforward_matrix.csv: matriz Q (1 coluna ABEV3).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.model.predictor import Predictor

PROJECT_ROOT = Path(__file__).resolve().parent
NEWS_PATH = PROJECT_ROOT / "intel" / "data acquisition" / "noticias_b3_sem_duplicados_sentimento.json"
TICKERS_DIR = PROJECT_ROOT / "tickers_data"
MODEL_DIR = PROJECT_ROOT / "modelo_7h_vwap"
LAG_LLAMA_DIR = PROJECT_ROOT / "lag-Llama"

OUT_LONG = PROJECT_ROOT / "q_ab_ev3_walkforward_long.csv"
OUT_MATRIX = PROJECT_ROOT / "q_ab_ev3_walkforward_matrix.csv"

SENTIMENT_MAP = {"positivo": 1.0, "negativo": -1.0, "neutro": 0.0}
TICKER = "ABEV3"
PREDICTION_LENGTH = 7
PERIOD_START = "202506"  # yyyymm
PERIOD_END = "202512"    # yyyymm
ALPHA = 0.75
B3_SESSION_HOURS = set(range(13, 20))  # 13h ... 19h => 7 barras horárias


def parse_news_datetime(data_str: str) -> datetime | None:
    if not data_str or not data_str.strip():
        return None
    try:
        return datetime.strptime(
            data_str.replace(" GMT", "").strip(),
            "%a, %d %b %Y %H:%M:%S",
        )
    except ValueError:
        return None


def load_daily_sentiment_for_ticker(path: Path, ticker: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de notícias não encontrado: {path}")

    rows = json.loads(path.read_text(encoding="utf-8"))
    parsed = []
    for item in rows:
        item_ticker = (item.get("ticker") or "").strip().upper()
        if not item_ticker or item_ticker != ticker:
            continue

        dt = parse_news_datetime(item.get("data") or "")
        if dt is None:
            janela_inicio = item.get("janela_inicio") or ""
            try:
                dt = datetime.strptime(janela_inicio, "%Y-%m-%d")
            except ValueError:
                continue

        sentimento = (item.get("sentimento") or "neutro").strip().lower()
        parsed.append(
            {
                "ticker": ticker,
                "view_date": pd.Timestamp(dt.date()),
                "sentimento_score": SENTIMENT_MAP.get(sentimento, 0.0),
            }
        )

    if not parsed:
        raise ValueError("Nenhuma notícia válida foi encontrada para agregar sentimento.")

    df = pd.DataFrame(parsed)
    grouped = (
        df.groupby(["ticker", "view_date"], as_index=False)
        .agg(
            sentiment_raw=("sentimento_score", "mean"),
            news_count=("sentimento_score", "size"),
        )
    )
    grouped["sentiment_signal"] = np.tanh(grouped["sentiment_raw"] * np.log1p(grouped["news_count"]))
    return grouped[["ticker", "view_date", "sentiment_signal", "news_count"]]


def month_bounds(period_start: str, period_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=int(period_start[:4]), month=int(period_start[4:]), day=1)
    end_month = pd.Timestamp(year=int(period_end[:4]), month=int(period_end[4:]), day=1)
    end = end_month + pd.offsets.MonthEnd(1)
    return start.normalize(), end.normalize()


def load_hourly_series_for_ticker(tickers_dir: Path, ticker: str) -> pd.DataFrame:
    file_path = tickers_dir / f"BMFBOVESPA_DLY_{ticker}, 60.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"CSV do ticker {ticker} não encontrado em: {file_path}")

    df = pd.read_csv(file_path, usecols=["time", "close", "Volume"])
    df["ticker"] = ticker
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("float32")
    df = df.dropna(subset=["time", "close", "Volume"]).sort_values("time").reset_index(drop=True)
    return df


def filter_b3_session(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém somente barras de pregão B3: dias úteis e janela 13h-19h."""
    out = df.copy()
    out["weekday"] = out["time"].dt.weekday
    out["hour"] = out["time"].dt.hour
    out = out[out["weekday"] <= 4]
    out = out[out["hour"].isin(B3_SESSION_HOURS)]
    out = out.sort_values("time").reset_index(drop=True)
    return out.drop(columns=["weekday", "hour"])


def add_vwap_target(df: pd.DataFrame, window_vwap: int = 7) -> pd.DataFrame:
    df = df.copy()
    df["pv"] = df["close"] * df["Volume"]
    df["vwap"] = df["pv"].rolling(window=window_vwap).sum() / df["Volume"].rolling(window=window_vwap).sum()
    df["vwap"] = df["vwap"].fillna(df["close"])
    df["log_ret_vwap"] = np.log(df["vwap"] / df["vwap"].shift(1))
    df = df.dropna(subset=["log_ret_vwap"]).copy()
    df["pv"] = df["pv"].astype("float32")
    df["vwap"] = df["vwap"].astype("float32")
    df["log_ret_vwap"] = df["log_ret_vwap"].astype("float32")

    # time_fake contínuo para o predictor (coerente com modelo_hibrido_tcc).
    df["time_fake"] = pd.date_range(start=df["time"].min(), periods=len(df), freq="h")
    df["view_date"] = df["time"].dt.normalize()
    return df


def load_predictor(model_dir: Path) -> Predictor:
    if LAG_LLAMA_DIR.exists():
        sys.path.insert(0, str(LAG_LLAMA_DIR))

    if not (model_dir / "gluonts-config.json").exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {model_dir}")
    try:
        return Predictor.deserialize(model_dir)
    except Exception as exc:
        raise RuntimeError(
            "Falha ao desserializar o modelo. Garanta que o pacote/código lag-llama "
            f"esteja disponível em {LAG_LLAMA_DIR} (ou instalado no ambiente). Detalhe: {exc}"
        ) from exc


def walk_forward_predict_7h(
    df: pd.DataFrame,
    predictor: Predictor,
    ticker: str,
    period_start: str,
    period_end: str,
) -> pd.DataFrame:
    start_date, end_date = month_bounds(period_start, period_end)
    daily_groups = df.groupby("view_date", sort=True)
    all_days = [d for d in sorted(daily_groups.groups.keys()) if start_date <= d <= end_date]

    results = []
    for day in all_days:
        day_df = daily_groups.get_group(day).sort_values("time")
        if len(day_df) != PREDICTION_LENGTH:
            # Ignora dias incompletos para manter equivalência 1 dia útil = 7h.
            continue
        if set(day_df["time"].dt.hour.tolist()) != B3_SESSION_HOURS:
            continue

        first_idx = int(day_df.index.min())
        context_df = df.iloc[:first_idx].copy()
        if len(context_df) < 60:
            continue
        context_df["log_ret_vwap"] = context_df["log_ret_vwap"].astype("float32")

        dataset_loop = PandasDataset.from_long_dataframe(
            context_df[["ticker", "time_fake", "log_ret_vwap"]],
            item_id="ticker",
            timestamp="time_fake",
            target="log_ret_vwap",
            freq="h",
        )
        forecast_it, _ = make_evaluation_predictions(
            dataset=dataset_loop,
            predictor=predictor,
            num_samples=10,
        )
        forecast = list(forecast_it)[0]

        pred_day_return = float(np.sum(forecast.mean[:PREDICTION_LENGTH]))
        real_day_return = float(day_df["log_ret_vwap"].sum())

        results.append(
            {
                "view_date": day,
                "ticker": ticker,
                "ret_7h_pred": pred_day_return,
                "ret_7h_real": real_day_return,
                "time_real_inicio": day_df["time"].iloc[0],
                "time_real_fim": day_df["time"].iloc[-1],
                "time_fake_inicio": day_df["time_fake"].iloc[0],
                "time_fake_fim": day_df["time_fake"].iloc[-1],
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "view_date",
                "ticker",
                "ret_7h_pred",
                "ret_7h_real",
                "time_real_inicio",
                "time_real_fim",
                "time_fake_inicio",
                "time_fake_fim",
                "ret_scale",
            ]
        )

    out = pd.DataFrame(results).sort_values("view_date").reset_index(drop=True)
    out["ret_scale"] = out["ret_7h_real"].abs().rolling(20, min_periods=5).mean().shift(1)
    fallback = out["ret_7h_real"].abs().median()
    if not np.isfinite(fallback) or fallback == 0:
        fallback = 0.005
    out["ret_scale"] = out["ret_scale"].fillna(fallback)
    return out


def combine_ai_and_sentiment(views: pd.DataFrame, sentiment: pd.DataFrame, alpha: float) -> pd.DataFrame:
    merged = views.merge(sentiment, on=["ticker", "view_date"], how="left")
    merged["sentiment_signal"] = merged["sentiment_signal"].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["sentiment_component"] = merged["sentiment_signal"] * merged["ret_scale"]
    merged["Q"] = alpha * merged["ret_7h_pred"] + (1.0 - alpha) * merged["sentiment_component"]
    return merged[
        [
            "view_date",
            "ticker",
            "Q",
            "ret_7h_pred",
            "ret_7h_real",
            "time_real_inicio",
            "time_real_fim",
            "time_fake_inicio",
            "time_fake_fim",
            "sentiment_signal",
            "sentiment_component",
            "news_count",
            "ret_scale",
        ]
    ].sort_values(["view_date", "ticker"]).reset_index(drop=True)


def main() -> None:
    print(f"Iniciando teste de Q com {TICKER} ({PERIOD_START} -> {PERIOD_END})...")

    print("Carregando sentimento diário...")
    sentiment = load_daily_sentiment_for_ticker(NEWS_PATH, ticker=TICKER)

    print("Carregando série horária do ativo...")
    prices = load_hourly_series_for_ticker(TICKERS_DIR, ticker=TICKER)
    prices = filter_b3_session(prices)
    prices = add_vwap_target(prices, window_vwap=PREDICTION_LENGTH)

    print("Carregando predictor Lag-Llama...")
    predictor = load_predictor(MODEL_DIR)

    print("Executando walk-forward (previsão 7h por dia)...")
    views = walk_forward_predict_7h(
        df=prices,
        predictor=predictor,
        ticker=TICKER,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )
    if views.empty:
        raise RuntimeError("Nenhuma view gerada no período. Revise período/modelo/série.")

    print("Combinando previsão + sentimento para formar Q...")
    q_long = combine_ai_and_sentiment(views, sentiment, alpha=ALPHA)
    q_matrix = q_long.pivot(index="view_date", columns="ticker", values="Q").sort_index()

    q_long["view_date"] = q_long["view_date"].dt.strftime("%Y-%m-%d")
    q_matrix.index = q_matrix.index.strftime("%Y-%m-%d")
    q_long.to_csv(OUT_LONG, index=False)
    q_matrix.to_csv(OUT_MATRIX)

    print(f"\nViews salvas em: {OUT_LONG}")
    print(f"Matriz Q salva em: {OUT_MATRIX}")
    print(f"Linhas (dias com previsão): {len(q_long)}")
    print(f"Dias com notícia: {(q_long['news_count'] > 0).sum()}")
    print(f"Acurácia direcional (pred vs real): {(np.sign(q_long['ret_7h_pred']) == np.sign(q_long['ret_7h_real'])).mean() * 100:.2f}%")
    print("\nÚltima linha da matriz Q:")
    print(q_matrix.tail(1).round(6))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
