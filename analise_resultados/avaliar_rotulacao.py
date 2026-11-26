"""Avalia apenas rótulos de consenso preenchidos por humanos."""
import json
import argparse
import hashlib
import shutil
from pathlib import Path
import pandas as pd
from .camadas_intermediarias import OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='CSV humano preenchido a importar apos validacao')
    args = parser.parse_args()
    target = OUT / 'rotulacao_humana.csv'
    if args.input:
        candidate = pd.read_csv(args.input).fillna('')
        reference = pd.read_csv(target).fillna('')
        if (len(candidate) != len(reference) or not candidate.sample_id.is_unique
                or set(candidate.sample_id) != set(reference.sample_id)):
            raise ValueError('Amostra ou identificadores diferentes da referencia.')
        a, b = candidate.set_index('sample_id').sort_index(), reference.set_index('sample_id').sort_index()
        for col in ('ticker', 'titulo', 'data'):
            if not a[col].equals(b[col]):
                raise ValueError(f'Conteudo da amostra alterado: {col}')
        for col in ('rotulo_avaliador_1', 'rotulo_avaliador_2', 'rotulo_consenso'):
            if not candidate[col].isin(['', 'positivo', 'negativo', 'neutro']).all():
                raise ValueError(f'Rotulos invalidos: {col}')
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        backup = OUT / f'rotulacao_humana_anterior_{digest[:12]}.csv'
        if not backup.exists():
            shutil.copyfile(target, backup)
        source_digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
        if args.input.resolve() != target.resolve():
            shutil.copyfile(args.input, target)
        (OUT / 'importacao_rotulacao.json').write_text(json.dumps(dict(
            source=str(args.input.resolve()), sha256=source_digest, n=len(candidate),
            identity_verified=['sample_id', 'ticker', 'titulo', 'data']), indent=2), encoding='utf-8')
    human = pd.read_csv(OUT / 'rotulacao_humana.csv').fillna('')
    model = pd.read_csv(OUT / 'gabarito_modelo.csv')
    if not human.sample_id.is_unique or not model.sample_id.is_unique or set(human.sample_id) != set(model.sample_id):
        raise ValueError('Identificadores humanos/modelo inconsistentes.')
    data = human.merge(model, on='sample_id', validate='one_to_one')
    labels = ['positivo', 'negativo', 'neutro']
    human_label = data.rotulo_consenso.str.strip().str.lower()
    invalid = ~human_label.isin(labels + [''])
    if invalid.any():
        raise ValueError('Rotulos invalidos: use positivo, negativo ou neutro.')
    data = data[human_label.isin(labels)].copy()
    data['human'] = human_label[human_label.isin(labels)]
    if data.empty:
        raise SystemExit('Pendente: nenhum rotulo humano de consenso preenchido. Nenhuma metrica estimada.')
    confusion = pd.crosstab(data.human, data.sentimento).reindex(index=labels, columns=labels, fill_value=0)
    confusion.to_csv(OUT / 'sentimento_matriz_confusao.csv')
    scores = []
    for label in labels:
        tp = int(confusion.loc[label, label])
        predicted, actual = int(confusion[label].sum()), int(confusion.loc[label].sum())
        scores.append(dict(label=label, support=actual, precision=tp / predicted if predicted else 0.,
                           recall=tp / actual if actual else None,
                           f1=2 * tp / (predicted + actual) if predicted + actual else None))
    pd.DataFrame(scores).to_csv(OUT / 'sentimento_metricas_humanas.csv', index=False)
    n = len(data)
    agreement = float((data.human == data.sentimento).mean())
    expected = float(sum(confusion.loc[k].sum() * confusion[k].sum() for k in labels) / n**2)
    macro_f1 = sum(s['f1'] for s in scores if s['f1'] is not None) / sum(s['f1'] is not None for s in scores)
    assessment = dict(n_evaluated=n, n_sample=len(human), partial=n < len(human),
        accuracy=agreement, matches=int((data.human == data.sentimento).sum()),
        disagreements=int((data.human != data.sentimento).sum()), macro_f1=macro_f1,
        cohen_kappa=(agreement-expected)/(1-expected) if expected < 1 else None,
        majority_human_baseline=float(data.human.value_counts().max()/n),
        human_raters=1 if human.rotulo_avaliador_2.eq('').all() else 2,
        reference='Single human annotation; not an adjudicated multi-rater gold standard.',
        source_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (OUT / 'sentimento_avaliacao.json').write_text(json.dumps(assessment, ensure_ascii=False, indent=2), encoding='utf-8')
    data[data.human != data.sentimento].to_csv(OUT / 'sentimento_divergencias.csv', index=False, encoding='utf-8-sig')
    pct = lambda v: f'{v*100:.2f}'.replace('.', ',')
    lines = [r'\subsection*{Avaliação humana da classificação de sentimento}',
        f'Foram avaliados {n} registros da amostra aleatória simples de 300 manchetes, selecionada com semente 2025. '
        'Um único avaliador humano classificou as manchetes sem visualizar os rótulos do modelo na interface. '
        'A coluna denominada consenso no formulário reproduz a avaliação individual; não representa consenso entre avaliadores. '
        'Os rótulos humanos são a referência operacional desta comparação, sujeitos a subjetividade.',
        r'\begin{table}[H]\centering\caption{Classificação do modelo em comparação com a referência humana.}',
        r'\begin{tabular}{lrrrr}\hline Classe & Suporte humano & Precisão (\%) & Revocação (\%) & F1 (\%) \\ \hline']
    for s in scores:
        lines.append(f"{s['label'].capitalize()} & {s['support']} & {pct(s['precision'])} & {pct(s['recall']) if s['recall'] is not None else '--'} & {pct(s['f1']) if s['f1'] is not None else '--'} " + r'\\')
    lines += [r'\hline\end{tabular}\par\smallskip{\footnotesize Fonte: elaboração própria.}\end{table}',
        f"A concordância observada é {pct(agreement)}\\% ({assessment['matches']} de {n}), com macro-F1 de {pct(macro_f1)}\\% "
        f"e coeficiente kappa de Cohen de {assessment['cohen_kappa']:.3f}. "
        f"A referência de sempre atribuir a classe humana mais frequente alcançaria {pct(assessment['majority_human_baseline'])}\\% de concordância nessa amostra.",
        r'\begin{table}[H]\centering\caption{Matriz de confusão do sentimento: linhas humanas e colunas do modelo.}',
        r'\begin{tabular}{lrrr}\hline Humano / Modelo & Positivo & Negativo & Neutro \\ \hline']
    for k in labels:
        lines.append(k.capitalize() + ' & ' + ' & '.join(str(int(confusion.loc[k,j])) for j in labels) + r'\\')
    lines += [r'\hline\end{tabular}\par\smallskip{\footnotesize Fonte: elaboração própria.}\end{table}',
        f"A principal divergência está na classe neutra: dos {int(confusion.loc['neutro'].sum())} registros assim classificados pelo humano, "
        f"o modelo atribui positivo a {int(confusion.loc['neutro', 'positivo'])} e negativo a {int(confusion.loc['neutro', 'negativo'])}. "
        f"O modelo utiliza a classe positiva em {int(confusion['positivo'].sum())} registros, frente a {int(confusion.loc['positivo'].sum())} na referência humana. "
        'Isso indica maior frequência de atribuição de polaridade pelo modelo nesta amostra, sobretudo positiva, em comparação com o critério do avaliador. '
        'O contraste sugere revisar os critérios de neutralidade e exemplos do prompt antes de uma nova rodada de classificação, '
        'cuja avaliação deve utilizar uma amostra separada para evitar adaptação à referência já observada.',
        'O kappa compara a concordância observada com aquela esperada pelas frequências marginais; não mede concordância entre dois avaliadores humanos. '
        'Divergências podem decorrer de erro do modelo ou de interpretação humana. Não foi mensurada a confiabilidade entre avaliadores humanos. '
        'A amostra abrange registros do corpus completo e não apenas notícias habilitadas no rebalanceamento de 2025. '
        'Registros de eventos relacionados podem ser dependentes. Esses indicadores avaliam a rotulação textual, sem demonstrar capacidade de antecipar retornos ou contribuição financeira do sentimento.']
    (OUT / 'sentimento_humano.tex').write_text('\n\n'.join(lines), encoding='utf-8')
    print(json.dumps(assessment, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
