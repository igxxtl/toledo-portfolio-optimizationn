"""Reproduz o backtest de 2025 com e sem a higienização das manchetes."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2] / 'GitHub' / 'toledo-portfolio-optimizationn'
sys.path.insert(0, str(REPO))

from intel.Blacklitterman import config as cfg, simular_dois_cenarios as engine
from intel.Blacklitterman.sentiment import parse_news_datetime, ticker_company_key
from intel.data_acquisition.config import NEWS_RAW, SENTIMENT_MODEL, SENTIMENT_MAX_WORKERS
from intel.data_acquisition.limpa_duplicados import normalize_title
from intel.data_acquisition.limpa_noticias import title_is_relevant


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def news_key(row):
    return row['ticker'].strip(), row['titulo'].strip()


def eligible_date(row):
    published = parse_news_datetime(row.get('data') or '')
    if published:
        return published.date()
    return datetime.fromisoformat(row['janela_inicio']).date()


def deduplicate(rows):
    # A rotina original preserva a primeira ocorrência global, antes do recorte de datas.
    seen = set()
    for row in rows:
        title = normalize_title(row.get('titulo') or '')
        if not title or title not in seen:
            yield row
        if title:
            seen.add(title)


def load_experiment():
    source = REPO / 'analise_resultados' / 'simulacao_pool'
    saved = pd.read_csv(source / 'auditoria_pool.csv', parse_dates=['signal_date', 'news_date'])
    days = set(saved.news_date.dt.date)
    companies = {ticker_company_key(asset) for asset in saved.asset}
    raw = read_json(NEWS_RAW)
    classified = read_json(cfg.NEWS_PATH)
    deduped = list(deduplicate(raw))
    relevant = [row for row in deduped if title_is_relevant(*news_key(row))]

    def eligible(rows):
        return [r for r in rows if ticker_company_key(r['ticker']) in companies and eligible_date(r) in days]

    # A base salva contém os rótulos originais e precisa ser reproduzida nos dias usados.
    expected = Counter((news_key(r), eligible_date(r)) for r in eligible(classified))
    assert expected == Counter((news_key(r), eligible_date(r)) for r in eligible(relevant))
    sets = {'com_filtro': eligible(classified), 'so_deduplicacao': eligible(deduped), 'sem_limpeza': eligible(raw)}
    known = {news_key(row): row['sentimento'] for row in classified}
    missing = sorted({news_key(r) for r in sets['sem_limpeza']} - known.keys())
    totals = {name: len(rows) for name, rows in zip(sets, (classified, deduped, raw))}
    return source, saved, sets, known, missing, totals


def classify_missing(missing):
    from dotenv import dotenv_values
    from openai import OpenAI
    from intel.data_acquisition.analise_sentimento import classify_headline

    key = os.environ.get('OPENAI_API_KEY') or dotenv_values(REPO / '.env').get('OPENAI_API_KEY')
    client = OpenAI(api_key=key, timeout=45, max_retries=1)

    def classify(pair):
        try:
            label = classify_headline(client, *pair).strip().lower()
        except Exception as exc:
            # O código usual cai para neutro em falhas; aqui exigimos um rótulo válido.
            raise RuntimeError(f'Classificador indisponível ({type(exc).__name__}, HTTP {getattr(exc, "status_code", None)})') from None
        if label not in cfg.SENTIMENT_MAP:
            raise ValueError('O classificador retornou um rótulo inválido.')
        return dict(ticker=pair[0], titulo=pair[1], sentimento=label)

    print(f'Classificando {len(missing)} títulos adicionais via {SENTIMENT_MODEL}.', flush=True)
    rows = [classify(missing[0])] if missing else []
    with ThreadPoolExecutor(max_workers=SENTIMENT_MAX_WORKERS) as executor:
        rows.extend(executor.map(classify, missing[1:]))
    (OUT / 'rotulos_adicionais.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


def run(source, saved, sets, known, missing, totals, neutral):
    if neutral:
        additional = dict.fromkeys(missing, 'neutro')
        suffix = '_adicionais_neutras'
    else:
        rows = read_json(OUT / 'rotulos_adicionais.json')
        additional = {news_key(row): row['sentimento'] for row in rows}
        assert additional.keys() == set(missing)
        assert set(additional.values()) <= cfg.SENTIMENT_MAP.keys()
        suffix = ''

    prices, returns, _ = engine.load_inputs()
    start, end = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')
    current, _, _ = engine.prepare_predictions(prices, start)
    pooled = pd.read_csv(source / 'previsoes_pool.csv', parse_dates=['date'])
    predictions = {}
    for asset, p in current.items():
        selected = pooled[(pooled.ticker == asset.removesuffix('.SA')) & (pooled.horizon == p['h'])]
        predictions[asset] = {**p, 'pred': selected.set_index('date').y_pred.reindex(prices.index)}

    labels = known | additional
    signals, audits = {}, {}
    loader = engine.load_daily_sentiment_for_ticker
    try:
        for scenario, rows in sets.items():
            path = OUT / f'entrada_{scenario}{suffix}.json'
            labeled = [{**r, 'sentimento': labels[news_key(r)]} for r in rows]
            path.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding='utf-8')
            engine.load_daily_sentiment_for_ticker = partial(loader, news_path=path)
            dates, signals[scenario], audits[scenario] = engine.build_signals(returns, prices, predictions, start, end, 126, 3.)
    finally:
        engine.load_daily_sentiment_for_ticker = loader

    match_cols = ['signal_date', 'news_date', 'asset', 'news_count', 'gate_active']
    baseline_audit = audits['com_filtro']
    pd.testing.assert_frame_equal(baseline_audit[match_cols], saved[match_cols])
    for col in ['Q', 'omega', 'prior', 'posterior']:
        np.testing.assert_allclose(baseline_audit[col], saved[col], rtol=1e-9, atol=1e-10)

    coverage = []
    for name, audit in audits.items():
        coverage.append(dict(cenario=name, registros_totais=totals[name],
                             manchetes_dias_elegiveis=len(sets[name]),
                             pares_com_noticia=int(audit.news_count.gt(0).sum()),
                             pares_gate_ativo=int(audit.gate_active.sum()),
                             pares_novos_gate=int((audit.gate_active & ~baseline_audit.gate_active).sum()),
                             pares_total=len(audit)))
    pd.DataFrame(coverage).to_csv(OUT / f'cobertura{suffix}.csv', index=False)
    pd.concat(audits, names=['cenario']).reset_index(level=0).to_csv(OUT / f'auditoria{suffix}.csv', index=False)

    archived = pd.read_csv(source / 'metricas.csv').query("model == 'BL_pool'").drop(columns='model')
    original_subsets = pd.read_csv(source / 'subconjuntos.csv')
    assert not archived.duplicated(['scenario', 'n', 'repetition']).any()
    results, trades = [archived.assign(cenario='com_filtro')], []
    comparison_metrics = ['return_pct', 'sharpe', 'max_drawdown_pct', 'cost_brl', 'final_capital_brl']
    same_signals = all(np.array_equal(signals['sem_limpeza'][d]['BL'], signals['so_deduplicacao'][d]['BL']) for d in dates if d in signals['sem_limpeza'])
    # O cenário filtrado já foi executado para os mesmos subgrupos e é validado abaixo.
    for n in range(1, 21):
        subsets = engine.subsets(n, len(returns.columns), 30, 2025)
        assert [';'.join(returns.columns[j] for j in s) for s in subsets] == (
            original_subsets.loc[original_subsets.n.eq(n)].sort_values('repetition').assets.tolist()
        )
        cases = [('mesmos_ativos', i, s) for i, s in enumerate(subsets)] + [('escolha_livre', 0, None)]
        current_rows = []
        simulated = None
        for cenario, sig in signals.items():
            if cenario == 'sem_limpeza' and same_signals and simulated is not None:
                current_rows.extend({**r, 'cenario': cenario} for r in simulated)
                trades.append(trades[-1].assign(cenario=cenario))
                continue
            arm_rows = []
            for scenario, rep, subset in cases:
                if cenario == 'com_filtro' and subset is not None:
                    continue
                metric, _, trade = engine.simulate(dates, returns, sig, n, 'BL', subset, 3., 10.)
                if cenario == 'com_filtro':
                    ref = archived.loc[archived.n.eq(n) & archived.scenario.eq(scenario)].iloc[0]
                    np.testing.assert_allclose([metric[k] for k in comparison_metrics],
                                               [ref[k] for k in comparison_metrics], rtol=1e-9, atol=1e-7)
                else:
                    arm_rows.append(dict(cenario=cenario, scenario=scenario, n=n, repetition=rep, **metric))
                if subset is None:
                    trades.append(trade.assign(cenario=cenario, n=n))
            if cenario == 'so_deduplicacao':
                simulated = arm_rows
            current_rows.extend(arm_rows)
        results.append(pd.DataFrame(current_rows))
        print(f'n={n}: comparação concluída.', flush=True)

    result = pd.concat(results, ignore_index=True)
    result.to_csv(OUT / f'resultados{suffix}.csv', index=False)
    pd.concat(trades, ignore_index=True).to_csv(OUT / f'rebalanceamentos{suffix}.csv', index=False)
    paired = result.pivot(index=['scenario', 'n', 'repetition'], columns='cenario', values='return_pct').reset_index()
    paired['diferenca_pp'] = paired.sem_limpeza - paired.com_filtro
    paired['diferenca_so_deduplicacao_pp'] = paired.so_deduplicacao - paired.com_filtro
    paired.to_csv(OUT / f'comparacao_pareada{suffix}.csv', index=False)
    print(pd.DataFrame(coverage).to_string(index=False))
    print(result.loc[result.scenario.eq('escolha_livre') & result.n.isin([5, 10, 20]),
                     ['cenario', 'n', *comparison_metrics]].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classificar', action='store_true', help='Classifica os títulos adicionais com o modelo do projeto.')
    parser.add_argument('--neutro', action='store_true', help='Controle que trata os títulos adicionais como neutros.')
    args = parser.parse_args()
    experiment = load_experiment()
    if args.classificar:
        classify_missing(experiment[4])
    else:
        run(*experiment, neutral=args.neutro)
