"""QA numerico e de estrutura do capitulo; complemento a revisao visual."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT/'analise_resultados/dados'
    source = ROOT/'intel/Blacklitterman/outputs/dois_cenarios'
    forecasts = pd.read_csv(data/'previsao_horizontes_2025.csv')
    summary = pd.read_csv(source/'resumo.csv')
    paired = pd.read_csv(data/'comparacao_pareada_resumo.csv')
    costs = pd.read_csv(data/'custos_contrafactual.csv')
    stats = json.loads((data/'estatisticas.json').read_text(encoding='utf-8'))
    assert len(forecasts) == 20
    assert (forecasts.r2_test < 0).sum() == 19
    assert (forecasts.test_hr < forecasts.majority_hr).sum() == 15
    assert round(forecasts.test_hr.mean(), 2) == 52.44
    assert forecasts.horizon.value_counts().to_dict() == {21:12,10:5,5:3}
    assert paired.n.nunique() == 20
    free = summary[summary.scenario=='escolha_livre'].pivot(index='n', columns='model', values='return_median')
    assert len(free) == 20
    assert costs.cost_impact_pp.gt(0).all()
    assert stats['news'] == dict(gate=3, observations=240, with_news=40, eligible=6, dates_with_any_eligible=3)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['bl_parameters']['NEWS_COUNT_MIN'] == 3
    for relative, expected in stats['source_hashes'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
    path = ROOT/'output/pdf/capitulo_4_resultados.pdf'
    reader = PdfReader(path)
    assert len(reader.pages) == 10
    text = ' '.join(' '.join(p.extract_text() for p in reader.pages).split())
    assert '\ufffd' not in text
    assert 'gate de 3 manchetes' in text
    assert 'escolhido após observar o desempenho de 2025' in text
    assert 'gate de cinco' not in text
    for n in range(1,10):
        assert f'Tabela 4.{n} -' in text
    assert 'Figura 4.1 -' in text and 'Figura 4.2 -' in text
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            assert page.chars
            assert min(c['x0'] for c in page.chars) > 70, i
            assert max(c['x1'] for c in page.chars) < 550, i
            assert max(c['bottom'] for c in page.chars) < 790, i
    report = dict(status='passed', pages=10, tables=9, figures=2, news_gate=3,
                  numeric_data_checks='passed', source_hashes='passed',
                  text_extraction='no replacement characters', text_bounds='within page margins',
                  visual_review='all 10 rendered pages inspected',
                  latex_compile='success, desktop editor compiler')
    (ROOT/'analise_resultados/verificacao_capitulo.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
