"""Auditoria e ablacao exploratoria reproduzivel, sem chamadas externas.

python -m analise_resultados.camadas_intermediarias
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from intel.Blacklitterman import simular_dois_cenarios as engine
from intel.Blacklitterman import config as cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analise_resultados/camadas'
VARIANTS = ('prior', 'xgb_gate', 'xgb', 'sentiment', 'hybrid')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices, returns, paths = engine.load_inputs()
    start, end = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')
    predictions, choices, pred_paths = engine.prepare_predictions(prices, start)
    choices.to_csv(OUT / 'horizontes.csv', index=False)
    metrics, audits, trades, curves = [], [], [], []
    for variant in VARIANTS:
        dates, signals, audit = engine.build_signals(returns, prices, predictions, start, end, 126, 3., variant)
        audits.append(audit)
        # n=20 isola alocacao; n=5 e 10 sao ilustrativos, nao tamanhos eleitos.
        for n in (5, 10, 20):
            cases = [('escolha_livre', 0, None)]
            cases += [('mesmos_ativos', i, s) for i, s in enumerate(engine.subsets(n, len(prices.columns), 30, 2025))]
            for scenario, repetition, subset in cases:
                m, curve, trade = engine.simulate(dates, returns, signals, n, 'BL', subset, 3., 10.)
                tags = dict(variant=variant, n=n, scenario=scenario, repetition=repetition)
                metrics.append({**tags, **m})
                trades.append(trade.assign(**tags))
                if scenario == 'escolha_livre':
                    curves.append(curve.assign(**tags))
        print(f'{variant}: concluido', flush=True)
    result = pd.DataFrame(metrics)
    result.to_csv(OUT / 'metricas.csv', index=False)
    pd.concat(audits).to_csv(OUT / 'auditoria.csv', index=False)
    pd.concat(trades).to_csv(OUT / 'pesos.csv', index=False)
    pd.concat(curves).to_csv(OUT / 'series_livres.csv', index=False)
    paired = result.pivot(index=['scenario', 'n', 'repetition'], columns='variant', values='return_pct')
    for variant in VARIANTS:
        paired[f'{variant}_minus_prior_pp'] = paired[variant] - paired['prior']
    paired.to_csv(OUT / 'comparacao_pareada.csv')
    # Controle de reproducao do motor original nas trajetorias livres.
    old = pd.read_csv(cfg.OUTPUTS_DIR / 'dois_cenarios/metricas.csv')
    old = old[(old.model == 'BL') & (old.scenario == 'escolha_livre')]
    new = result[(result.variant == 'hybrid') & (result.scenario == 'escolha_livre')]
    check = new.merge(old, on='n', suffixes=('_new', '_old'))
    difference = float((check.return_pct_new - check.return_pct_old).abs().max())
    if len(check) != 3 or difference > 1e-7:
        raise AssertionError(f'Hibrido difere da rodada original: {difference}')
    corpus = pd.DataFrame(json.loads(cfg.NEWS_PATH.read_text(encoding='utf-8')))
    corpus.groupby(['ticker', 'sentimento']).size().rename('quantidade').to_csv(OUT / 'distribuicao_sentimento.csv')
    # Amostra aleatoria simples: rótulo humano fica vazio; modelo em arquivo separado.
    sample = corpus.sample(n=min(300, len(corpus)), random_state=2025).reset_index(drop=True)
    sample.insert(0, 'sample_id', range(1, len(sample) + 1))
    sample[['sample_id', 'sentimento']].to_csv(OUT / 'gabarito_modelo.csv', index=False, encoding='utf-8-sig')
    columns = [c for c in ('sample_id', 'ticker', 'titulo', 'data') if c in sample]
    blind = sample[columns].copy()
    blind['rotulo_avaliador_1'] = ''
    blind['rotulo_avaliador_2'] = ''
    blind['rotulo_consenso'] = ''
    blind['observacao'] = ''
    # Nao sobrescrever anotacoes de uma execucao anterior.
    if not (OUT / 'rotulacao_humana.csv').exists():
        blind.to_csv(OUT / 'rotulacao_humana.csv', index=False, encoding='utf-8-sig')
    paths += pred_paths + [Path(__file__)]
    manifest = dict(exploratory=True, seed=2025, cardinalities=[5, 10, 20],
                    gate=cfg.NEWS_COUNT_MIN, hybrid_reproduction_max_difference_pp=difference,
                    sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    variants={
                        'prior': 'Posterior exatamente igual ao prior, sem views.',
                        'xgb_gate': 'Retira sentimento de Q; preserva gate e confianca hibrida.',
                        'xgb': 'Sem gate de noticias; confianca apenas do modelo.',
                        'sentiment': 'Gate original; confianca textual; escala e horizonte herdados do XGB.',
                        'hybrid': 'Especificacao original sem alteracao.'},
                    limitations=['Ablacao retrospectiva em 2025; nao constitui teste independente.',
                                 'Variante textual ainda herda horizonte/escala quantitativos.',
                                 'XGB livre muda confianca e gate; nao e remocao isolada do sentimento.',
                                 'Dependencia entre subconjuntos e alvos sobrepostos; sem testes iid.'])
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [r'\section{Ablação e diagnóstico das camadas intermediárias}',
             'A análise complementar é retrospectiva e utiliza o mesmo motor, custos de 10 bps e rebalanceamento mensal. '
             'As cardinalidades 5, 10 e 20 são ilustrativas. Não houve nova seleção de hiperparâmetros.',
             r'\begin{table}[H]\centering\caption{Ablação com seleção própria em 2025.}',
             r'\begin{tabular}{lrrrr}\hline Variante & n & Retorno (\%) & Sharpe & MDD (\%) \\ \hline']
    labels = dict(prior='Sem views', xgb_gate='XGB com gate', xgb='XGB livre', sentiment='Textual', hybrid='Híbrido')
    for r in result[result.scenario == 'escolha_livre'].itertuples():
        lines.append(f'{labels[r.variant]} & {r.n} & {r.return_pct:.2f} & {r.sharpe:.2f} & {r.max_drawdown_pct:.2f} '+r'\\')
    lines += [r'\hline\end{tabular}\par\smallskip{\footnotesize Fonte: elaboração própria.}\end{table}',
              'A variante XGB com gate preserva a confiança híbrida e remove somente a parcela textual de Q. '
              'Sua comparação com o híbrido avalia o efeito incremental dessa parcela sob a mesma regra de cobertura. '
              'XGB livre remove também a dependência de notícias e usa confiança quantitativa. '
              'A variante textual utiliza confiança de notícias, mas ainda herda escala e horizonte quantitativos; '
              'portanto, não representa um sistema textual inteiramente independente.',
              'Nos conjuntos comuns, as comparações são pareadas por subconjunto; os dados completos estão nos artefatos de auditoria. '
              'Com vinte ativos, elimina-se a diferença de elegibilidade entre as estratégias. '
              'O resultado descreve o efeito das intervenções neste motor e neste período, sem demonstrar generalização futura.']
    (OUT / 'ablacao.tex').write_text('\n\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
