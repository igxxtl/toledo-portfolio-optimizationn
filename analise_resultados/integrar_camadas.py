"""Integra fragmentos gerados ao TCC local, sem duplicar secoes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .camadas_intermediarias import OUT


def main():
    tcc = Path(r'C:\Users\igor.toledo_visagio\Documents\TCC 2')
    metrics = pd.read_csv(OUT / 'metricas.csv')
    audit = pd.read_csv(OUT / 'auditoria.csv')
    x = pd.read_csv(OUT / 'xgb_baselines_trimestres.csv')
    annual = x[x.period == '2025'].pivot(index='asset', columns='model', values='rmse')
    wins_zero, wins_hist = int((annual.xgb < annual.zero).sum()), int((annual.xgb < annual.historical).sum())
    # Distancia total dos pesos-alvo nos mesmos casos e datas de execucao.
    weights = pd.read_csv(OUT / 'pesos.csv')
    key = ['scenario', 'n', 'repetition', 'execution_date']
    base = weights[weights.variant == 'prior'][key + ['weights']].rename(columns={'weights': 'prior_weights'})
    distance = weights.merge(base, on=key, validate='many_to_one')
    def l1(row):
        a, b = json.loads(row.weights), json.loads(row.prior_weights)
        return .5 * sum(abs(a.get(k, 0.) - b.get(k, 0.)) for k in a.keys() | b.keys())
    distance['half_l1_target_vs_prior'] = distance.apply(l1, axis=1)
    distance.drop(columns=['weights', 'prior_weights']).to_csv(OUT / 'distancia_pesos_prior.csv', index=False)
    summary = audit.groupby('variant').agg(n=('asset', 'size'), eligible=('gate_active', 'sum'),
        mean_abs_posterior_shift=('posterior_minus_prior', lambda s: s.abs().mean()),
        max_abs_posterior_shift=('posterior_minus_prior', lambda s: s.abs().max()))
    summary.to_csv(OUT / 'resumo_influencia.csv')
    # Exemplos deterministas: sem noticia, gate ativo, discordancia de sinais.
    hybrid = audit[audit.variant == 'hybrid']
    cases = [hybrid[hybrid.news_count == 0].head(1), hybrid[hybrid.gate_active].head(1),
             hybrid[hybrid.gate_active & (np.sign(hybrid.prediction) != np.sign(hybrid.sentiment_signal))].head(1)]
    pd.concat(cases).drop_duplicates(['signal_date', 'asset']).to_csv(OUT / 'exemplos_cadeia.csv', index=False)
    # Foco solicitado: XGB em fragmento proprio; avaliacao humana fica somente no apoio.
    lines = []
    lines.append((OUT / 'ablacao.tex').read_text(encoding='utf-8'))
    lines += ['O resultado mais relevante é a proximidade entre o híbrido e o prior sem views com vinte ativos: '
        '28,2085\\% e 28,2053\\% de retorno líquido, respectivamente, diferença de apenas 0,0032 ponto percentual. '
        'Com seleção própria de cinco e dez ativos, o prior retorna 59,24\\% e 42,63\\%, frente a 58,40\\% e 40,14\\% do híbrido. '
        'Não se observa ganho consistente de retorno ao adicionar as views ao prior nessas trajetórias.',
        'Ao manter o gate e a confiança híbrida, mas retirar o sentimento de Q, os retornos passam a 40,26\\%, 33,78\\% e 20,30\\% para cinco, dez e vinte ativos. '
        'A parcela textual melhora o resultado frente a esse controle, mas isso não demonstra superioridade frente à ausência de views. '
        'O XGB sem gate apresenta retornos negativos nas três cardinalidades; essa variante altera também a confiança e deve ser interpretada como intervenção conjunta.',
        'A cobertura de seis pares habilitados permanece uma medida de elegibilidade, inclusive nas tabelas de auditoria das variantes sem gate. '
        'O afastamento entre posterior e prior e a distância entre pesos-alvo foram exportados para auditoria. '
        'A meia distância L1 entre pesos-alvo não equivale ao turnover realizado, pois ignora a deriva dos pesos entre negociações.',
        'Esses resultados reforçam que a vantagem frente a Markowitz não pode ser atribuída automaticamente ao aprendizado de máquina ou ao sentimento. '
        'O prior, a estrutura de covariância e as restrições respondem por parte substancial do comportamento observado. '
        'A interpretação continua limitada por escolhas retrospectivas e por apenas um ano de avaliação financeira.']
    fragment = '\n\n'.join(lines) + '\n'
    (OUT / 'camadas_intermediarias.tex').write_text(fragment, encoding='utf-8')
    dest = tcc / '05_Resultados/camadas_intermediarias.tex'
    dest.write_text(fragment, encoding='utf-8')
    chapter = tcc / '05_Resultados/resultados.tex'
    text = chapter.read_text(encoding='utf-8')
    include = r'\input{05_Resultados/camadas_intermediarias}'
    if include not in text:
        text = text.replace(r'\section{Limitações e alcance das conclusões}', include + '\n\n' + r'\section{Limitações e alcance das conclusões}')
    chapter.write_text(text, encoding='utf-8')
    print('Fragmento integrado; resultados e distancias exportados.')


if __name__ == '__main__':
    main()
