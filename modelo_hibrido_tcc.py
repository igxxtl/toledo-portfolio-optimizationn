# -*- coding: utf-8 -*-
"""
Modelo Híbrido TCC – versão para uso local no projeto.

Leitura dos dados: pasta tickers_data (CSVs no formato BMFBOVESPA_DLY_TICKER, 60.csv).
Sem dependência de Google Drive ou Colab.

Pré-requisitos (fora do script):
  - pip install gluonts ujson pandas numpy torch lightning scikit-learn matplotlib seaborn
  - lag-Llama: clone em projeto com  git clone https://github.com/time-series-foundation-models/lag-Llama/

Checkpoint: o script baixa automaticamente lag-llama.ckpt para checkpoints/ se não existir.
  URL: https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

# --- Configuração de caminhos (projeto local) ---
PROJECT_ROOT = Path(__file__).resolve().parent
TICKERS_DATA_DIR = PROJECT_ROOT / "tickers_data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LAG_LLAMA_DIR = PROJECT_ROOT / "lag-Llama"

CHECKPOINT_URL = "https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt"
CHECKPOINT_FILENAME = "lag-llama.ckpt"
MODELO_PASTA = "modelo_7h_vwap"


def baixar_checkpoint_se_necessario() -> Path:
    """
    Cria a pasta checkpoints/ se não existir e baixa lag-llama.ckpt do Hugging Face
    caso o arquivo ainda não esteja lá.
    Retorna o path do arquivo (checkpoints/lag-llama.ckpt).
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / CHECKPOINT_FILENAME

    if ckpt_path.exists():
        print(f"Checkpoint já existe: {ckpt_path}")
        return ckpt_path

    print(f"Baixando checkpoint de {CHECKPOINT_URL} ...")
    try:
        urlretrieve(CHECKPOINT_URL, ckpt_path)
        print(f"Checkpoint salvo em: {ckpt_path}")
        return ckpt_path
    except Exception as e:
        raise RuntimeError(
            f"Falha ao baixar o checkpoint. Baixe manualmente e coloque em {ckpt_path}\n"
            f"  URL: {CHECKPOINT_URL}\n  Erro: {e}"
        ) from e


def carregar_dados_tickers(pasta: Path | None = None) -> pd.DataFrame:
    """Carrega todos os CSVs da pasta tickers_data e concatena com coluna ticker."""
    pasta = pasta or TICKERS_DATA_DIR
    if not pasta.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta}")

    csv_files = list(pasta.glob("*.csv"))
    dataframes = []

    for file_path in csv_files:
        try:
            file_name = file_path.name
            # Remove prefixo 'BMFBOVESPA_DLY_' e pega o ticker até a vírgula
            ticker = file_name.replace("BMFBOVESPA_DLY_", "").split(",")[0].strip()
            df = pd.read_csv(file_path)
            df["ticker"] = ticker
            dataframes.append(df)
        except Exception as e:
            print(f"Erro ao ler {file_path}: {e}")

    if not dataframes:
        raise ValueError("Nenhum CSV válido encontrado em tickers_data.")

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Tickers carregados: {combined_df['ticker'].unique().tolist()}")
    return combined_df


def preparar_dataset(df: pd.DataFrame, data_min: str = "2021-01-01") -> pd.DataFrame:
    """Converte tempo, tipos, ordena e filtra por data; calcula log_return, log_volume, spreads."""
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    cols_numericas = ["open", "high", "low", "close", "Volume"]
    df[cols_numericas] = df[cols_numericas].astype("float32")
    df = df.sort_values(["ticker", "time"]).reset_index(drop=True)
    df = df[df["time"] >= data_min].copy()

    df["log_return"] = df.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    df["log_volume"] = df.groupby("ticker")["Volume"].transform(lambda x: np.log(x / x.shift(1)))
    df["spread_hl"] = np.log(df["high"] / df["low"])
    df["spread_co"] = np.log(df["close"] / df["open"])
    df = df.dropna()

    return df


def main():
    # 1. Leitura a partir de tickers_data
    combined_df = carregar_dados_tickers()
    combined_df = preparar_dataset(combined_df)

    # 2. Preparação para o modelo (ex.: ABEV3)
    from gluonts.dataset.pandas import PandasDataset

    df_abev = combined_df[combined_df["ticker"] == "ABEV3"].copy()

    # Divisão temporal
    df_train = df_abev[df_abev["time"] < "2026-01-01"].copy()
    df_test_input = df_abev[
        (df_abev["time"] >= "2026-01-01") & (df_abev["time"] <= "2026-02-20")
    ].copy()
    df_gabarito = df_abev[df_abev["time"] > "2026-02-20"].copy()

    # time_fake contínuo para GluonTS
    full_range = pd.date_range(start=df_abev["time"].min(), periods=len(df_abev), freq="h")
    df_abev["time_fake"] = full_range
    df_train = df_abev.loc[df_train.index]
    df_test_input = df_abev.loc[df_test_input.index]
    df_gabarito = df_abev.loc[df_gabarito.index]

    dataset_train_puro = PandasDataset.from_long_dataframe(
        df_train, item_id="ticker", timestamp="time_fake", target="close", freq="h"
    )
    dataset_test_puro = PandasDataset.from_long_dataframe(
        df_test_input, item_id="ticker", timestamp="time_fake", target="close", freq="h"
    )

    print(f"Treino: {len(df_train)} registros. Teste (input): {len(df_test_input)}. Gabarito: {len(df_gabarito)}.")

    # 3. Lag-Llama: garantir que o módulo está no path
    if LAG_LLAMA_DIR.exists():
        import sys
        sys.path.insert(0, str(LAG_LLAMA_DIR))

    caminho_modelo = PROJECT_ROOT / MODELO_PASTA
    modelo_ja_salvo = (caminho_modelo / "gluonts-config.json").exists()

    if modelo_ja_salvo:
        print(f"Modelo já existe em {caminho_modelo}. Carregando para previsão (sem treinar).")
        from gluonts.model.predictor import Predictor
        predictor_puro = Predictor.deserialize(caminho_modelo)
    else:
        ckpt_path = baixar_checkpoint_se_necessario()

        import torch
        import gluonts.torch.modules.loss as gt_loss
        import gluonts.torch.distributions.studentT as gt_studentT
        torch.serialization.add_safe_globals([
            gt_loss.NegativeLogLikelihood,
            gt_studentT.StudentTOutput,
        ])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        prediction_length = 7

        from lightning.pytorch.callbacks import EarlyStopping
        from lag_llama.gluon.estimator import LagLlamaEstimator

        early_stop = EarlyStopping(monitor="train_loss", patience=5, verbose=True, mode="min")
        estimator = LagLlamaEstimator(
            prediction_length=prediction_length,
            context_length=32,
            input_size=1,
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args.get("time_fake", True),
            batch_size=32,
            trainer_kwargs={
                "max_epochs": 50,
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1,
                "callbacks": [early_stop],
            },
        )

        predictor_puro = estimator.train(dataset_train_puro)

    # 4. VWAP e log_ret_vwap para avaliação
    df_abev["pv"] = df_abev["close"] * df_abev["Volume"]
    window_vwap = 7
    df_abev["vwap"] = (
        df_abev["pv"].rolling(window=window_vwap).sum()
        / df_abev["Volume"].rolling(window=window_vwap).sum()
    )
    df_abev["vwap"] = df_abev["vwap"].fillna(df_abev["close"])
    df_abev["log_ret_vwap"] = np.log(df_abev["vwap"] / df_abev["vwap"].shift(1))
    df_abev["log_ret_vwap"] = df_abev["log_ret_vwap"].astype("float32")
    df_abev["close"] = df_abev["close"].astype("float32")
    df_abev.dropna(subset=["log_ret_vwap"], inplace=True)

    # 5. Walk-forward 7h
    from gluonts.evaluation import make_evaluation_predictions

    prediction_length = 7
    resultados_7h = []
    df_analise = df_abev[df_abev["time"] >= "2026-01-01"].copy()
    df_historico_base = df_abev[df_abev["time"] < "2026-01-01"].copy()
    posicao_atual = 0
    total_linhas_teste = len(df_analise)

    while posicao_atual + prediction_length <= total_linhas_teste:
        df_contexto = pd.concat([df_historico_base, df_analise.iloc[:posicao_atual]])
        dataset_loop = PandasDataset.from_long_dataframe(
            df_contexto,
            item_id="ticker",
            timestamp="time_fake",
            target="log_ret_vwap",
            freq="h",
        )
        forecast_it, _ = make_evaluation_predictions(
            dataset=dataset_loop,
            predictor=predictor_puro,
            num_samples=100,
        )
        forecast = list(forecast_it)[0]
        ultimo_preco_real = df_contexto["close"].iloc[-1]
        df_gabarito_bloco = df_analise.iloc[
            posicao_atual : posicao_atual + prediction_length
        ]
        preco_ref = ultimo_preco_real

        for i in range(prediction_length):
            ret_ia = forecast.mean[i]
            preco_ia = preco_ref * np.exp(ret_ia)
            preco_ref = preco_ia
            resultados_7h.append({
                "time_real": df_gabarito_bloco["time"].iloc[i],
                "hora_previsao": i + 1,
                "ret_ia": ret_ia,
                "ret_real": df_gabarito_bloco["log_ret_vwap"].iloc[i],
                "previsao_ia_reais": preco_ia,
                "valor_real_reais": df_gabarito_bloco["close"].iloc[i],
            })
        posicao_atual += prediction_length

    df_final_7h = pd.DataFrame(resultados_7h)
    acerto_direcao = np.mean(
        np.sign(df_final_7h["ret_real"]) == np.sign(df_final_7h["ret_ia"])
    ) * 100
    mae_final = np.abs(
        df_final_7h["previsao_ia_reais"] - df_final_7h["valor_real_reais"]
    ).mean()

    print("\n=== RESULTADOS: MODELO 7 HORAS (VWAP) ===")
    print(f"Acurácia direcional: {acerto_direcao:.2f}%")
    print(f"MAE médio: R$ {mae_final:.3f}")

    from sklearn.metrics import mean_absolute_error, r2_score

    y_real = df_final_7h["valor_real_reais"]
    y_pred = df_final_7h["previsao_ia_reais"]
    mape_global = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    print(f"MAE: R$ {mean_absolute_error(y_real, y_pred):.3f}")
    print(f"R²: {r2_score(y_real, y_pred):.4f}")
    print(f"MAPE: {mape_global:.2f}%")

    # 6. Salvar modelo só se acabou de treinar (não sobrescreve se foi carregado)
    if not modelo_ja_salvo:
        caminho_modelo.mkdir(parents=True, exist_ok=True)
        try:
            predictor_puro.serialize(caminho_modelo)
            print(f"Modelo salvo em: {caminho_modelo}")
        except Exception as e:
            print(f"Erro ao salvar modelo: {e}")
    else:
        print(f"Modelo já existente em {caminho_modelo} (não sobrescrito).")

    # 7. Opcional: gerar zip do modelo (sem download do Colab)
    import shutil
    nome_zip = "modelo_7h_vwap_completo"
    zip_path = PROJECT_ROOT / nome_zip
    shutil.make_archive(str(zip_path), "zip", str(caminho_modelo))
    print(f"Zip do modelo: {zip_path}.zip")

    # 8. Gráfico de comparação (salvo em arquivo)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 7))
        plt.plot(
            df_final_7h["time_real"],
            df_final_7h["valor_real_reais"],
            label="Real (B3)",
            color="#2c3e50",
            alpha=0.6,
            linewidth=1.5,
        )
        plt.plot(
            df_final_7h["time_real"],
            df_final_7h["previsao_ia_reais"],
            label="IA (7h VWAP)",
            color="#16a085",
            linewidth=1.2,
        )
        plt.title("ABEV3: Validação Walk-Forward 7 Horas (VWAP)", fontsize=14)
        plt.ylabel("Preço (R$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig_path = PROJECT_ROOT / "validacao_7h_vwap.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"Gráfico salvo em: {fig_path}")
    except Exception as e:
        print(f"Gráfico não gerado: {e}")

    return combined_df, df_final_7h, predictor_puro


if __name__ == "__main__":
    main()
