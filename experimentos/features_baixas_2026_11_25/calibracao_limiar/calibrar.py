"""Calibra em 2024 o limiar direcional do beta de baixa e o aplica em 2025."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'previsoes.csv'


def evaluate(predictions, threshold):
    frame = predictions.assign(
        real=np.sign(predictions.actual).astype(int),
        classe=np.sign(predictions.previsao - threshold).astype(int),
    )
    rows = []
    for ticker, group in frame.groupby('ticker'):
        up = group[group.real.eq(1)]
        down = group[group.real.eq(-1)]
        assert len(up) and len(down)
        rows.append(dict(ticker=ticker, altas=len(up), baixas=len(down), neutros=int(group.real.eq(0).sum()),
                         altas_corretas=int(up.classe.eq(1).sum()), baixas_corretas=int(down.classe.eq(-1).sum()),
                         acerto_altas=100 * up.classe.eq(1).mean(),
                         acerto_baixas=100 * down.classe.eq(-1).mean()))
    result = pd.DataFrame(rows)
    result['acuracia_balanceada'] = (result.acerto_altas + result.acerto_baixas) / 2
    return result


def calibrate(validation, target_up):
    # Pontos médios representam todas as classificações possíveis sem empatar no limiar.
    scores = np.sort(validation.previsao.unique())
    candidates = np.r_[np.nextafter(scores[0], -np.inf),
                       scores[:-1] + (scores[1:] - scores[:-1]) / 2,
                       np.nextafter(scores[-1], np.inf)]
    up = validation[validation.actual.gt(0)]
    weights = 1 / up.groupby('ticker').ticker.transform('size').to_numpy()
    rates = 100 * (weights @ (up.previsao.to_numpy()[:, None] > candidates)) / up.ticker.nunique()
    feasible = rates >= target_up - 1e-10
    threshold = float(candidates[feasible][-1])
    assert feasible.any() and evaluate(validation, threshold).acerto_altas.mean() >= target_up - 1e-10
    return threshold, len(candidates)


def main():
    thesis = OUT.parents[2]
    originals = [SOURCE, *thesis.glob('**/*.tex')]
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in originals}
    forecasts = pd.read_csv(SOURCE, parse_dates=['date', 'maturity'])
    scope = forecasts[forecasts.dias_corridos.gt(1) & forecasts.modelo.isin(['referencia', 'beta_baixas_63'])]
    key = ['ticker','date','maturity','horizonte','fase','actual']
    assert scope.horizonte.eq(21).all()
    assert scope.groupby(key).modelo.nunique().eq(2).all()
    assert not scope.duplicated(key + ['modelo']).any()
    validation = scope[scope.fase.eq('validation')]
    target_up = evaluate(validation[validation.modelo.eq('referencia')], 0).acerto_altas.mean()
    threshold, n_candidates = calibrate(validation[validation.modelo.eq('beta_baixas_63')], target_up)

    arms = [('referencia', 'referencia', 0.), ('beta_zero', 'beta_baixas_63', 0.),
            ('beta_calibrado', 'beta_baixas_63', threshold)]
    parts = [evaluate(group[group.modelo.eq(model)], cutoff).assign(fase=phase, variante=label, limiar=cutoff)
             for phase, group in scope.groupby('fase') for label, model, cutoff in arms]
    by_asset = pd.concat(parts, ignore_index=True)
    assert by_asset.groupby(['fase','variante']).ticker.nunique().eq(20).all()
    columns = ['acerto_altas','acerto_baixas','acuracia_balanceada']
    summary = by_asset.groupby(['fase','variante']).agg(acoes=('ticker','size'),
        altas=('altas','sum'), baixas=('baixas','sum'), neutros=('neutros','sum'),
        altas_corretas=('altas_corretas','sum'), baixas_corretas=('baixas_corretas','sum'),
        limiar=('limiar','first'), **{name: (name,'mean') for name in columns}).reset_index()
    by_asset.to_csv(OUT / 'metricas_por_acao.csv', index=False)
    summary.to_csv(OUT / 'resumo.csv', index=False)
    assert all(sha256(path.read_bytes()).hexdigest() == value for path, value in hashes.items())
    (OUT / 'protocolo.json').write_text(json.dumps(dict(horizonte_pregoes=21, recorte='dias_corridos > 1',
        calibracao='2024', avaliacao='2025', meta_acerto_altas=target_up,
        limiar_log_retorno=threshold, limiares_avaliados=n_candidates,
        arquivos_conferidos=len(hashes), sha256_previsoes=hashes[SOURCE]), indent=2), encoding='utf-8')
    print(summary.round(6).to_string(index=False))


if __name__ == '__main__':
    main()
