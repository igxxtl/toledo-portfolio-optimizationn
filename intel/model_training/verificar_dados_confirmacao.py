"""Solicita precos posteriores ao acervo, em pasta isolada e apos congelar a receita."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analise_resultados/evolucao_xgb'


def main():
    freeze=json.loads((OUT/'receita_congelada.json').read_text(encoding='utf-8'))
    for p,h in freeze['sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Receita mudou antes da consulta')
    status=dict(requested_window=freeze['proposed_window'],source='Yahoo Finance via yfinance',
                recipe_frozen_before_request=True,original_files_changed=False)
    try:
        import yfinance as yf
        yf.set_tz_cache_location(str(ROOT/'tmp/yfinance_cache'))
        sample=yf.download('ABEV3.SA',start='2026-03-01',end='2026-11-09',auto_adjust=True,
                           progress=False,threads=False,timeout=12)
        if sample.empty:
            status.update(status='unavailable',reason='O provedor retornou amostra vazia para a janela solicitada.')
        else:
            if isinstance(sample.columns,pd.MultiIndex):sample.columns=sample.columns.get_level_values(0)
            sample.index=pd.to_datetime(sample.index).tz_localize(None)
            available=sample[sample.index>=pd.Timestamp(freeze['proposed_window'][0])]
            folder=OUT/'novos_dados';folder.mkdir(exist_ok=True)
            sample.to_csv(folder/'ABEV3_consulta.csv')
            status.update(status='available' if len(available) else 'unavailable',rows_after_old_dataset=len(available),
                          first=str(sample.index.min().date()),last=str(sample.index.max().date()))
    except Exception as e:
        status.update(status='unavailable',reason=type(e).__name__+': '+str(e))
    (OUT/'disponibilidade_confirmacao.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
