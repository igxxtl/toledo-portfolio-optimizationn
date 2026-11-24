"""Benchmark temporal reproduzível de retornos da B3."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

SEED = 42
HORIZONS = (1, 5, 10, 21)
CANDIDATES = ((2, 120), (3, 220), (5, 300))
QUARTERS = pd.date_range("2025-01-01", periods=5, freq="QS")
VALID_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-12-31")


def load_data(repo, limit, tabular=True):
    files = sorted((repo / "dados_diarios").glob("*.csv"))
    files = [f for f in files if f.stem not in {"IBOV", "USDBRL", "DI"}]
    if tabular:
        sys.path.insert(0, str(repo))
        from intel.model_training import train_xgb as original
        ranks = original.build_cross_sectional_ranks(files)
    data = {}
    for path in files[:limit]:
        if tabular:
            raw = original.carregar_df(path)
        else:
            raw = pd.read_csv(path, usecols=["date", "close"], parse_dates=["date"]).sort_values("date")
        raw = raw.loc[raw.date.le(TEST_END)].reset_index(drop=True)
        if raw.date.duplicated().any() or not raw.close.gt(0).all():
            raise ValueError(f"Invalid date or close in {path}")
        X = None
        if tabular:
            for col in ranks[path.stem]:
                raw[col] = ranks[path.stem][col].reindex(raw.date).to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                features, columns = original.build_features(raw)
            X = features[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        dates = raw.date.to_numpy()
        close = raw.close.to_numpy()
        for h in HORIZONS:
            raw[f"y{h}"] = np.log(raw.close.shift(-h) / raw.close)
        data[path.stem] = (dates, close, X, raw[[f"y{h}" for h in HORIZONS]].to_numpy())
    return data


def splits(dates, h, start, end):
    i = np.arange(len(dates) - h)
    train = i[dates[i + h] < np.datetime64(start)]
    test = i[(dates[i] >= np.datetime64(start)) & (dates[i] < np.datetime64(end))]
    assert train[-1] + h < test[0]
    return train[train >= 252], test


def rows(ticker, dates, h, y, indices, model, pred, phase):
    return pd.DataFrame(dict(ticker=ticker, date=dates[indices],
                             target_date=dates[indices + h], horizon=h, y=y[indices],
                             model=model, pred=pred, phase=phase))


def tree_job(item):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier, XGBRegressor

    ticker, (dates, close, X, Y), h = item
    y = Y[:, HORIZONS.index(h)]
    output = []
    def fit(train, test, name, candidate):
        X_train, X_test = X[train], X[test]
        yy = y[train]
        if name == "Zero":
            return np.zeros(len(test))
        if name == "Media historica":
            return np.full(len(test), yy.mean())
        if name == "Ridge":
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100))
            return model.fit(X_train, yy).predict(X_test)
        depth, rounds = candidate
        params = dict(max_depth=depth, n_estimators=rounds, learning_rate=.035, subsample=.8,
                      colsample_bytree=.8, min_child_weight=12, reg_lambda=5,
                      tree_method="hist", random_state=SEED, n_jobs=1)
        weight = np.exp(np.linspace(-3, 0, len(train)))
        if name == "XGBoost regressao":
            reg = XGBRegressor(**params, objective="reg:squarederror").fit(X_train, yy, sample_weight=weight)
            return reg.predict(X_test)
        q = np.quantile(yy, [1/3, 2/3])
        labels = np.digitize(yy, q)
        averages = np.array([yy[labels == k].mean() for k in range(3)])
        clf = XGBClassifier(**params, objective="multi:softprob").fit(X_train, labels, sample_weight=weight)
        return clf.predict_proba(X_test) @ averages

    versions = ("XGBoost regressao", "XGBoost 3 classes")
    val_train, val = splits(dates, h, VALID_START, QUARTERS[0])
    val = val[dates[val + h] < np.datetime64(QUARTERS[0])]
    choices = {}
    for name in versions:
        options = [fit(val_train, val, name, candidate) for candidate in CANDIDATES]
        scores = [np.mean(np.sign(p) == np.sign(y[val])) for p in options]
        rmse = [np.sqrt(np.mean((p - y[val]) ** 2)) for p in options]
        best = min(range(len(options)), key=lambda k: (-scores[k], rmse[k]))
        choices[name] = (CANDIDATES[best], float(scores[best]), float(rmse[best]))
        output.append(rows(ticker, dates, h, y, val, name, options[best], "validacao"))
    selected = min(choices, key=lambda name: (-choices[name][1], choices[name][2]))
    for start, end in zip(QUARTERS[:-1], QUARTERS[1:]):
        train, test = splits(dates, h, start, end)
        preds = {name: fit(train, test, name, choices[name][0] if name in choices else None)
                 for name in ("Zero", "Media historica", "Ridge", *versions)}
        preds["XGBoost projeto"] = preds[selected]
        for name, pred in preds.items():
            output.append(rows(ticker, dates, h, y, test, name, pred, "teste"))
    return output, dict(ticker=ticker, horizonte=h, variante=selected, candidatos=choices)


def tree_run(data, out):
    jobs = [(ticker, series, h) for ticker, series in data.items() for h in HORIZONS]
    frames, choices = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for i, (results, choice) in enumerate(pool.map(tree_job, jobs), start=1):
            frames.extend(results)
            choices.append(choice)
            if i % 10 == 0 or i == len(jobs):
                print(f"Tabulares: {i}/{len(jobs)} combinações ativo-horizonte", flush=True)
    pd.concat(frames).to_csv(out / "previsoes_tabulares.csv", index=False)
    (out / "selecao_tabular.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")


def lightgbm_job(item):
    from lightgbm import LGBMRegressor

    ticker, (dates, _, X, Y), h = item
    y = Y[:, HORIZONS.index(h)]
    output = []
    for start, end in zip(QUARTERS[:-1], QUARTERS[1:]):
        train, test = splits(dates, h, start, end)
        model = LGBMRegressor(n_estimators=150, learning_rate=.035, num_leaves=7,
                              max_depth=3, max_bin=63, min_child_samples=20,
                              subsample=.8, subsample_freq=1, colsample_bytree=.8,
                              reg_lambda=5, verbosity=-1, n_jobs=1, random_state=SEED,
                              deterministic=True, force_col_wise=True)
        model.fit(X[train], y[train], sample_weight=np.exp(np.linspace(-3, 0, len(train))))
        pred = model.booster_.predict(X[test], num_threads=1)
        output.append(rows(ticker, dates, h, y, test, "LightGBM", pred, "teste"))
    return output


def lightgbm_run(data, out):
    jobs = [(ticker, series, h) for ticker, series in data.items() for h in HORIZONS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        frames = [frame for results in pool.map(lightgbm_job, jobs) for frame in results]
    pd.concat(frames).to_csv(out / "previsoes_lightgbm.csv", index=False)


def gru_run(data, out):
    import torch
    from torch import nn

    torch.set_num_threads(3)
    output = []
    lookback = 35
    for h in HORIZONS:
        for quarter, (start, end) in enumerate(zip(QUARTERS[:-1], QUARTERS[1:])):
            torch.manual_seed(SEED)
            rng = np.random.default_rng(SEED)
            fit_x, fit_y, fit_id, stop_x, stop_y, stop_id, tasks = [], [], [], [], [], [], []
            for k, (ticker, (dates, _, X, Y)) in enumerate(data.items()):
                y = Y[:, HORIZONS.index(h)]
                train, test = splits(dates, h, start, end)
                stop = train[int(.85 * len(train)):]
                core = train[train + h < stop[0]]
                by_pos = np.arange(lookback)[None, :] - lookback + 1
                columns = np.isfinite(X[core]).any(axis=0)
                Z = X.copy()
                Z[:, ~columns] = 0
                med = np.nanmedian(Z[core], axis=0)
                Z = np.where(np.isfinite(Z), Z, med)
                Z = np.clip((Z - Z[core].mean(axis=0)) / np.maximum(Z[core].std(axis=0), 1e-6), -10, 10)
                mu, sd = y[core].mean(), y[core].std()
                for ix, xx, yy, ids in [(core, fit_x, fit_y, fit_id), (stop, stop_x, stop_y, stop_id)]:
                    xx.append(Z[ix[:, None] + by_pos])
                    yy.append((y[ix] - mu) / sd)
                    ids.append(np.full(len(ix), k))
                tasks.append((ticker, dates, y, test, Z[test[:, None] + by_pos], k, mu, sd))

            def tensors(x, y, ids):
                return (torch.tensor(np.concatenate(x), dtype=torch.float32),
                        torch.tensor(np.concatenate(y), dtype=torch.float32),
                        torch.tensor(np.concatenate(ids), dtype=torch.long))

            Xtr, Ytr, Itr = tensors(fit_x, fit_y, fit_id)
            Xva, Yva, Iva = tensors(stop_x, stop_y, stop_id)

            class GRU(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gru = nn.GRU(Xtr.shape[-1], 16, batch_first=True)
                    self.ticker = nn.Embedding(len(data), 4)
                    self.head = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Dropout(.30), nn.Linear(32, 1))
                def forward(self, x, ids):
                    hidden = self.gru(x)[1][-1]
                    return self.head(torch.cat([hidden, self.ticker(ids)], 1)).squeeze(-1)

            model = GRU()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
            best, best_epoch = float("inf"), -1
            state = None
            for epoch in range(18):
                model.train()
                order = rng.permutation(len(Ytr))
                for b in range(0, len(order), 512):
                    ix = order[b:b+512]
                    optimizer.zero_grad()
                    loss = nn.functional.mse_loss(model(Xtr[ix], Itr[ix]), Ytr[ix])
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    p = torch.cat([model(x, i) for x, i in zip(Xva.split(512), Iva.split(512))])
                    score = nn.functional.mse_loss(p, Yva).item()
                if score < best - 1e-5:
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best, best_epoch = score, epoch
                if epoch - best_epoch >= 4:
                    break
            assert state is not None
            model.load_state_dict(state)
            model.eval()
            with torch.inference_mode():
                for ticker, dates, y, test, xt, k, mu, sd in tasks:
                    pred = model(torch.tensor(xt, dtype=torch.float32),
                                 torch.full((len(test),), k)).numpy() * sd + mu
                    output.append(rows(ticker, dates, h, y, test, "GRU", pred, "teste"))
            print(f"GRU h={h} trimestre={quarter+1} epoca={best_epoch+1}", flush=True)
    pd.concat(output).to_csv(out / "previsoes_gru.csv", index=False)


def lag_run(data, repo, lag_path, out):
    from types import MethodType
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    sys.path.insert(0, str(lag_path))
    from gluonts.dataset.common import ListDataset
    from lag_llama.gluon.estimator import LagLlamaEstimator

    checkpoint = repo / "checkpoints" / "lag-llama.ckpt"
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checksum != "b5a5c4b8a0cfe9b81bdac35ed5d88b5033cd119b5206c28e9cd67c4b45fb2c96":
        raise ValueError("Checkpoint diferente do checkpoint público Lag-Llama esperado")
    # The historical checkpoint is a PyTorch Lightning bundle from the pinned project commit.
    import os
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kw = ckpt["hyper_parameters"]["model_kwargs"]
    estimator = LagLlamaEstimator(
        prediction_length=max(HORIZONS), context_length=32, input_size=kw["input_size"],
        n_layer=kw["n_layer"], n_embd_per_head=kw["n_embd_per_head"], n_head=kw["n_head"],
        scaling=kw["scaling"], time_feat=kw["time_feat"], ckpt_path=str(checkpoint),
        batch_size=96, num_parallel_samples=1, device=torch.device("cpu"),
        lags_seq=["D"], use_single_pass_sampling=True,
    )
    # Use the exact numerical offsets in the checkpoint, avoiding alias changes
    # between pandas releases for quarter-end and month-end frequencies.
    estimator.lags_seq = list(kw["lags_seq"])
    module = estimator.create_lightning_module(use_kv_cache=True)

    def recursive_mean(self, past_target, past_observed_values,
                       past_time_feat=None, future_time_feat=None):
        forecasts = []
        for t in range(self.prediction_length):
            params, loc, scale = self.model(
                past_target=past_target, past_observed_values=past_observed_values,
                past_time_feat=past_time_feat,
                future_time_feat=future_time_feat[..., :t+1, :] if future_time_feat is not None else None,
                use_kv_cache=self.use_kv_cache,
            )
            mean = self.model.distr_output.distribution([p[:, -1:] for p in params], loc, scale).mean
            past_target = torch.cat((past_target, mean), dim=1)
            past_observed_values = torch.cat((past_observed_values, torch.ones_like(mean)), dim=1)
            forecasts.append(mean)
        self.model.reset_cache()
        return torch.cat(forecasts, dim=1).unsqueeze(1)

    module.forward = MethodType(recursive_mean, module)
    predictor = estimator.create_predictor(estimator.create_transformation(), module)
    output = []
    for ticker, (dates, close, X, Y) in data.items():
        ret = np.diff(np.log(close))
        all_idx = np.flatnonzero(dates >= np.datetime64(QUARTERS[0]))
        all_idx = all_idx[all_idx < len(dates)-1]
        max_past = 1200
        dataset = ListDataset([dict(start=pd.Timestamp(dates[max(0, i-max_past)+1]),
                                    target=ret[max(0, i-max_past):i])
                               for i in all_idx], freq="B")
        predictions = np.array([f.samples[0] for f in predictor.predict(dataset)])
        assert predictions.shape == (len(all_idx), max(HORIZONS)) and np.isfinite(predictions).all()
        for col, h in enumerate(HORIZONS):
            keep = all_idx + h < len(dates)
            output.append(rows(ticker, dates, h, Y[:, col], all_idx[keep],
                               "Lag-Llama", np.cumsum(predictions, axis=1)[keep, h-1], "teste"))
        print(f"Lag-Llama {ticker}: {len(all_idx)} origens", flush=True)
    (out / "lag_checkpoint.json").write_text(json.dumps({
        "sha256": checksum,
        "model_kwargs": {k: str(v) for k, v in kw.items()},
    }, indent=2), encoding="utf-8")
    pd.concat(output).to_csv(out / "previsoes_lag.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--lag-source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=("tree", "gru", "lag", "lightgbm"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    data = load_data(args.repo.resolve(), args.limit, tabular=args.engine != "lag")
    print(f"{args.engine}: {len(data)} ativos", flush=True)
    if args.engine == "tree":
        tree_run(data, args.out)
    elif args.engine == "lightgbm":
        lightgbm_run(data, args.out)
    elif args.engine == "gru":
        gru_run(data, args.out)
    else:
        lag_run(data, args.repo.resolve(), args.lag_source.resolve(), args.out)


if __name__ == "__main__":
    main()
