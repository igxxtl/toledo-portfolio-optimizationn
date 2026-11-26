"""Consolida auditoria, features e metodologias sem atualizar modelos publicados."""
from pathlib import Path
import json
import os
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'analise_resultados'
OUT=BASE/'evolucao_xgb'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    features=BASE/'features_xgb';method=BASE/'metodologia_xgb'
    fs=pd.read_csv(features/'resumo_horizontes.csv');ms=pd.read_csv(method/'resumo_horizontes.csv')
    fc=pd.read_csv(features/'comparacao_2025.csv');mc=pd.read_csv(method/'comparacao_2025.csv')
    fr=json.loads((features/'resultado.json').read_text());mr=json.loads((method/'resultado.json').read_text())
    assert len(fc)==len(mc)==60
    comparison=mc.merge(fc[['ticker','horizon','rmse_anchor']],on=['ticker','horizon'],validate='one_to_one')
    comparison.to_csv(OUT/'comparacao_final_por_ativo.csv',index=False)
    combined=ms.copy();combined.to_csv(OUT/'resumo_final.csv',index=False)
    # Alinhamento das previsoes de todas as etapas, mantendo os horizontes atuais.
    choices=pd.read_csv(method/'escolhas_2024.csv').set_index(['ticker','horizon']).chosen
    observations=pd.read_csv(method/'previsoes.csv')
    selected=[]
    for (ticker,h),g in observations[observations.phase.eq('test')].groupby(['ticker','horizon']):
        chosen=choices.loc[(ticker,h)]
        selected.append(g[['ticker','horizon','date','maturity','actual','current','previous','feature_selected',chosen]].rename(
            columns={chosen:'candidate'}).assign(chosen_method=chosen))
    selected=pd.concat(selected);selected.to_csv(OUT/'previsoes_finais_2025.csv',index=False)
    # Efeito puro do metodo versus expansivo, com o mesmo conjunto de features por horizonte.
    fixed=pd.read_csv(method/'metricas.csv');fixed=fixed[fixed.phase.eq('test')]
    effects=[]
    for (ticker,h),g in fixed.groupby(['ticker','horizon']):
        g=g.set_index('model');anchor=g.loc['expansivo']
        for model in ['movel_2a','movel_4a','vol_normalizada','pool','pool_vol']:
            r=g.loc[model]
            effects.append(dict(ticker=ticker,horizon=h,method=model,
                rmse_reduction_pct=(1-r.rmse/anchor.rmse)*100,hr_gain_pp=r.hr-anchor.hr))
    effects=pd.DataFrame(effects);effects.to_csv(OUT/'efeitos_metodologia.csv',index=False)
    effects.groupby(['horizon','method'],as_index=False).agg(
        mean_rmse_reduction_pct=('rmse_reduction_pct','mean'),median_rmse_reduction_pct=('rmse_reduction_pct','median'),
        wins=('rmse_reduction_pct',lambda s:int((s>0).sum()))).to_csv(OUT/'efeitos_metodologia_resumo.csv',index=False)
    pooled=[]
    for h,g in fixed.groupby('horizon'):
        pivot=g.set_index(['ticker','model'])
        p=pivot.xs('pool',level='model');old=pivot.xs('current',level='model')
        zero=pivot.xs('zero',level='model');history=pivot.xs('historical',level='model')
        pooled.append(dict(horizon=int(h),rmse_current=float(old.rmse.mean()),rmse_pool=float(p.rmse.mean()),
            mae_current=float(old.mae.mean()),mae_pool=float(p.mae.mean()),hr_current=float(old.hr.mean()),hr_pool=float(p.hr.mean()),
            rmse_reduction_pct=float((1-p.rmse.mean()/old.rmse.mean())*100),
            wins_current=int((p.rmse<old.rmse).sum()),wins_zero=int((p.rmse<zero.rmse).sum()),wins_historical=int((p.rmse<history.rmse).sum())))
    pooled=pd.DataFrame(pooled);pooled.to_csv(OUT/'modelo_conjunto_exploratorio.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/xgb_matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    stages=[('rmse_current','Atual'),('rmse_previous','Ajustes anteriores'),('rmse_features','Features'),('rmse_new','Metodologia')]
    x=np.arange(3);width=.18
    for j,(col,label) in enumerate(stages):axes[0].bar(x+(j-1.5)*width,ms[col]*100,width,label=label)
    axes[0].set_xticks(x,[f'{int(h)} pregões' for h in ms.horizon]);axes[0].set_ylabel('RMSE × 100 (log-retorno)')
    axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.15)
    for j,(col,label) in enumerate([('mae_current','Atual'),('mae_new','Metodologia')]):
        axes[1].bar(x+(j-.5)*.3,ms[col]*100,.3,label=label)
    axes[1].set_xticks(x,[f'{int(h)} pregões' for h in ms.horizon]);axes[1].set_ylabel('MAE × 100 (log-retorno)');axes[1].legend(fontsize=8)
    fig.suptitle('Evolução das previsões · mesmas observações de 2025\nReceitas escolhidas na validação anterior, sem reescolha pelo teste')
    fig.tight_layout();fig.savefig(OUT/'evolucao_erros.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    for j,(column,label) in enumerate([('rmse_current','Atual'),('rmse_pool','Modelo conjunto fixo')]):
        axes[0].bar(x+(j-.5)*.34,pooled[column]*100,.34,label=label)
    axes[0].set_ylabel('RMSE × 100 (log-retorno)');axes[0].legend(fontsize=8)
    for column,label in [('hr_current','Atual'),('hr_pool','Modelo conjunto fixo')]:
        axes[1].plot(x,pooled[column],marker='o',label=label)
    axes[1].set_ylabel('Acurácia direcional (%)');axes[1].set_ylim(0,100);axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,[f'{int(h)} pregões' for h in pooled.horizon]);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Modelo conjunto fixo · resultado exploratório em 2025\nDestaque posterior ao teste; requer confirmação em período novo')
    fig.tight_layout();fig.savefig(OUT/'modelo_conjunto_exploratorio.png',dpi=180);plt.close(fig)
    pivot=mc.pivot(index='ticker',columns='horizon',values='rmse_reduction_pct')
    fig,ax=plt.subplots(figsize=(7,8));limit=max(float(abs(pivot.to_numpy()).max()),1.)
    im=ax.imshow(pivot,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
    ax.set_xticks(range(3),['5 pregões','10 pregões','21 pregões']);ax.set_yticks(range(20),pivot.index)
    for i in range(20):
        for j in range(3):
            v=pivot.iloc[i,j];ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.55*limit else '#182b35')
    ax.set_title('Redução de RMSE da candidata final contra o atual (%)\nPositivo: melhora · Negativo: piora')
    fig.colorbar(im,ax=ax,label='Redução do RMSE (%)');fig.tight_layout();fig.savefig(OUT/'melhoria_final_por_ativo.png',dpi=180);plt.close(fig)
    fmt=lambda v:f'{v:.2f}'.replace('.',',')
    audit=pd.read_csv(features/'redundancia_original.csv');prices=pd.read_csv(features/'auditoria_precos.csv')
    lines=['EVOLUÇÃO DA PREVISÃO XGBOOST — AUDITORIA, FEATURES E METODOLOGIA','',
        'RESUMO',
        'Foram preservados os modelos, as previsões e os resultados financeiros atuais. Esta rodada não substitui o pipeline nem altera o TCC automaticamente. '
        'O alvo continua sendo o retorno numérico, avaliado nos mesmos ativos, horizontes e datas do diagnóstico atual.',
        'As configurações foram escolhidas por RMSE ou erro relativo na validação de 2024; o ranking de 2025 não foi usado para substituir essas escolhas. '
        'Como 2025 já foi examinado e os diagnósticos orientaram esta pesquisa, todos os resultados de 2025 permanecem retrospectivos.','',
        '1. AUDITORIA DE DADOS E FEATURES',
        f'Arquivos auditados: {len(prices)}; não foram encontrados fechamentos não positivos, datas duplicadas ou ordenação inválida. '
        f'Nos vinte ativos, {len(audit)} pares de features têm Spearman absoluto superior a 0,98; '
        f'{int(audit.duplicate_numeric.sum())} pares são numericamente duplicados, contados por ativo.',
        'Há uma linha com abertura/máxima/mínima zeradas por ação em 08/04/2026, fora das janelas de treino e teste. '
        'Esses campos foram tratados como ausentes na construção das novas features, sem alterar os CSVs originais. '
        'O volume zero do câmbio não é interpretado como falha de liquidez de uma ação. '
        'A auditoria de consistência OHLC não comprova a ausência de revisões históricas ou a correção de todos os ajustes por eventos corporativos.','',
        '2. EXPERIMENTO DE FEATURES',
        'Referência: mesmas 103 features, regressão quadrática com parâmetros fixos e pesos iguais. '
        'Conjuntos novos: enxuto (24 variáveis); enxuto com retornos curtos, candles, gaps, volatilidade e risco específico; '
        'e ampliado com medidas transversais de mercado. Os filtros de correlação/constância/ausência usam somente o fit purgado, sem y e sem o teste.',
        'O conjunto por ativo/horizonte foi escolhido em 2024. Não foi feita uma nova busca de hiperparâmetros para cada conjunto, '
        'para manter a comparação das entradas controlada. '
        f'Escolhas: {json.dumps(fr["chosen_counts"],ensure_ascii=False)}.',
        f'Em 2025, a escolha de features melhora o atual em {fr["wins_current"]}/60 combinações, '
        f'mas melhora a referência de parâmetros fixos em apenas {fr["wins_anchor"]}/60. '
        'Logo, parte da diferença contra o atual é do protocolo de treinamento e não pode ser atribuída às novas features.','',
        '3. EXPERIMENTO DE METODOLOGIA',
        'Comparados: individual expansivo; janelas móveis de dois e quatro anos; alvo normalizado pela volatilidade conhecida; '
        'modelo conjunto dos vinte ativos com identidade da ação; modelo conjunto com normalização por volatilidade. '
        'No alvo normalizado, a previsão é reconstruída em retorno antes da avaliação, preservando a pergunta original.',
        'Para comparar modelos individuais e conjuntos, usa-se um conjunto comum de features por horizonte, escolhido '
        'pelo score médio relativo ao retorno zero de cada ativo em 2024. '
        'Essa escolha global difere da seleção individual da etapa de features. '
        'Os contrastes contra o expansivo dentro da etapa de metodologia usam o mesmo conjunto comum, '
        'e distinguem esse efeito do simples contraste entre etapas.',
        'Os modelos conjuntos aprendem de todos os ativos com o mesmo horizonte, ordenados por data; '
        'há purga por maturidade também na validação interna. A identidade é conhecida previamente e não usa resultados futuros. '
        'A seleção final do método para cada ativo/horizonte usa apenas os resultados de 2024.',
        f'Métodos escolhidos: {json.dumps(mr["chosen_counts"],ensure_ascii=False)}.','',
        '4. RESULTADO PAREADO EM 2025',
        'Todos os erros abaixo são médias simples entre os vinte ativos, em log-retorno multiplicado por 100. '
        'Comparar atual e candidata dentro do mesmo horizonte; não misturar os níveis de erro de prazos diferentes.']
    for r in ms.itertuples():
        lines += [f'{int(r.horizon)} pregões: MAE {fmt(r.mae_current*100)} → {fmt(r.mae_new*100)}; '
            f'RMSE {fmt(r.rmse_current*100)} → {fmt(r.rmse_new*100)}; redução relativa {fmt(r.rmse_reduction_pct)}%; '
            f'HR {fmt(r.hr_current)}% → {fmt(r.hr_new)}%; melhora contra o atual em {int(r.wins_current)}/20 ativos.']
    c=mr['current_horizons']
    lines += [f'Nos horizontes atualmente usados pelo sistema: RMSE médio {fmt(c["rmse_current"]*100)} → {fmt(c["rmse_new"]*100)}; '
        f'HR {fmt(c["hr_current"])}% → {fmt(c["hr_new"])}%. '
        'Essa média reúne horizontes diferentes por ativo e é somente uma comparação auxiliar com composição idêntica.',
        f'A candidata supera o modelo atual em {mr["wins_current"]}/60 casos, a rodada de hiperparâmetros em {mr["wins_previous"]}/60, '
        f'a candidata da etapa de features em {mr["wins_features"]}/60, o retorno zero em {mr["wins_zero"]}/60 '
        f'e a média histórica observável em {mr["wins_historical"]}/60.','',
        'RESULTADO EXPLORATÓRIO DO MODELO CONJUNTO FIXO',
        'O modelo pool foi uma das seis metodologias predefinidas. Como configuração fixa, mostrou resultados mais promissores em 2025. '
        'O destaque abaixo foi motivado pela avaliação de 2025 e não altera as escolhas anteriores feitas em 2024. '
        'Ele deve ser confirmado em um período novo antes de qualquer promoção ao pipeline.']
    for r in pooled.itertuples():
        lines.append(f'{r.horizon} pregões: MAE {fmt(r.mae_current*100)} → {fmt(r.mae_pool*100)}; '
            f'RMSE {fmt(r.rmse_current*100)} → {fmt(r.rmse_pool*100)}; redução {fmt(r.rmse_reduction_pct)}%; '
            f'HR {fmt(r.hr_current)}% → {fmt(r.hr_pool)}%; melhora contra o atual em {r.wins_current}/20, '
            f'contra zero em {r.wins_zero}/20 e contra média histórica em {r.wins_historical}/20.')
    lines += ['',
        '5. CONFIRMAÇÃO E USO NO BL',
        'O acervo atual já foi utilizado em diagnósticos até 2026, incluindo a avaliação OOF completa. '
        'Não há neste acervo um bloco inteiramente intocado que possa ser chamado de confirmação independente. '
        'É necessário congelar uma receita e obter/avaliar dados de um período ainda não examinado. '
        'Nenhum resultado retrospectivo é apresentado como prova confirmatória.',
        'Reduzir erro contra o modelo antigo não basta: deve haver melhora contra referências simples, consistência entre ativos '
        'e avaliação em outra janela. Ainda que uma candidata melhore o erro, seu uso no BL e sua contribuição financeira precisam '
        'de uma comparação separada, mantendo prior, confiança, custos e restrições. '
        'Os modelos e o backtest atuais permanecem como referência.','',
        '6. REPRODUÇÃO',
        'python -m unittest intel.model_training.test_features_experimento',
        'python -m intel.model_training.experimento_features --audit-only',
        'python -m intel.model_training.experimento_features',
        'python -m intel.model_training.experimento_metodologia',
        'python -m analise_resultados.resumir_features_metodologia',
        'Dependências e hashes estão nos protocolos de cada etapa. Os dados originais não foram alterados; a tentativa de obter cotações novas é registrada separadamente.']
    availability=OUT/'disponibilidade_confirmacao.json'
    if availability.exists():
        status=json.loads(availability.read_text(encoding='utf-8'))
        lines+=['','Consulta para confirmação: '+json.dumps(status,ensure_ascii=False),
            'A receita foi congelada antes de solicitar preços de 09/04/2026 a 08/11/2026. '
            'A consulta ao provedor falhou com timeout de resolução de rede e não forneceu cotações utilizáveis. '
            'A confirmação independente e a avaliação financeira da nova candidata permanecem pendentes. '
            'Não se alterou a base original nem se substituiu o modelo de produção.']
    (OUT/'RELATORIO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Relatorio e figuras consolidados.')


if __name__=='__main__':main()
