"""Gera capitulo PDF e fragmento LaTeX a partir dos CSVs auditados."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import shutil
import numpy as np
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Line, String, PolyLine, Polygon
from reportlab.graphics import renderPDF, renderSVG

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'analise_resultados'
DATA = WORK / 'dados'
SOURCE = ROOT / 'intel/Blacklitterman/outputs/dois_cenarios'
PDFOUT = ROOT / 'output/pdf'
TEXOUT = Path(r'C:\Users\igor.toledo_visagio\Desktop\TCC\05_Resultados')
WIDTH = A4[0] - 5 * cm
BLUE, ORANGE = colors.HexColor('#2166ac'), colors.HexColor('#c95722')


def fmt(x, digits=2):
    numeric = float(x)
    if round(numeric, digits) == 0:
        numeric = 0.
    return f'{numeric:,.{digits}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def line_chart(name, panels, bands=False, capital=False):
    """Figura vetorial a partir dos dados, sem interpolacao estatistica."""
    d = Drawing(WIDTH, 220)
    panel_width = (WIDTH - 32) / len(panels)
    for idx, (title, lines) in enumerate(panels):
        x0, y0, w, h = 32 + idx * panel_width, 50, panel_width - 35, 140
        values = [float(v) for label, xs, ys, color, lo, hi in lines for v in ys]
        if bands:
            values += [float(v) for label, xs, ys, color, lo, hi in lines for vs in (lo, hi) if vs is not None for v in vs]
        ymin, ymax = min(values), max(values)
        pad = max((ymax - ymin) * .08, 1.)
        ymin, ymax = ymin - pad, ymax + pad
        allx = [float(v) for label, xs, ys, color, lo, hi in lines for v in xs]
        xmin, xmax = min(allx), max(allx)
        xx = lambda v: x0 + (v - xmin) / max(xmax - xmin, 1) * w
        yy = lambda v: y0 + (v - ymin) / (ymax - ymin) * h
        d.add(String(x0 + w / 2, 207, title, textAnchor='middle', fontName='TimesB', fontSize=10))
        for value in np.linspace(ymin, ymax, 5):
            d.add(Line(x0, yy(value), x0 + w, yy(value), strokeColor=colors.HexColor('#dddddd'), strokeWidth=.4))
            label = f'{value / 1000:.0f}' if capital else f'{value:.0f}'
            d.add(String(x0 - 5, yy(value) - 3, label, textAnchor='end', fontName='Times', fontSize=8))
        for label, xs, ys, color, lo, hi in lines:
            if bands and lo is not None:
                points = [c for x,y in list(zip(xs,lo)) + list(reversed(list(zip(xs,hi)))) for c in (xx(x),yy(y))]
                tint = colors.Color(color.red * .18 + .82, color.green * .18 + .82, color.blue * .18 + .82)
                d.add(Polygon(points, fillColor=tint, strokeColor=None))
        for label, xs, ys, color, lo, hi in lines:
            points = [c for x,y in zip(xs,ys) for c in (xx(x),yy(y))]
            dash = [4,3] if label == 'Selic' else None
            d.add(PolyLine(points, strokeColor=color, strokeWidth=1.3, strokeDashArray=dash))
        ticks = [1,5,10,15,20] if not capital else [0,62,125,187,249]
        for value in ticks:
            d.add(String(xx(value), y0 - 13, str(value + 1) if capital else str(value),
                         textAnchor='middle', fontName='Times', fontSize=8))
        d.add(String(x0 + w/2, 7, 'Pregões (capital em R$ mil)' if capital else 'Número de ativos (n)',
                     textAnchor='middle', fontName='Times', fontSize=9))
    d.add(String(32, 24, 'BL: azul | Markowitz: laranja | Selic: cinza', fontName='Times', fontSize=8))
    renderPDF.drawToFile(d, str(TEXOUT / f'figuras/{name}.pdf'))
    renderSVG.drawToFile(d, str(TEXOUT / f'figuras/{name}.svg'))
    return d


def latex(s):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '²': r'$^{2}$', '³': r'$^{3}$', 'Ω': r'$\Omega$',
                    'π': r'$\pi$', 'α': r'$\alpha$', 'τ': r'$\tau$',
                    'Σ': r'$\Sigma$', 'Δ': r'$\Delta$', '≤': r'$\leq$', '≥': r'$\geq$',
                    '×': r'$\times$'}
    return ''.join(replacements.get(c,c) for c in s)


def main():
    for directory in (PDFOUT, TEXOUT, TEXOUT/'figuras'):
        directory.mkdir(parents=True, exist_ok=True)
    for name, fn in [('Times','times.ttf'),('TimesB','timesbd.ttf'),('TimesI','timesi.ttf')]:
        pdfmetrics.registerFont(TTFont(name, f'C:/Windows/Fonts/{fn}'))
    body = ParagraphStyle('body', fontName='Times', fontSize=12, leading=18, alignment=TA_JUSTIFY,
                          firstLineIndent=1.25*cm, spaceAfter=7)
    small = ParagraphStyle('small', fontName='Times', fontSize=9.5, leading=12.3, spaceAfter=7, alignment=TA_LEFT)
    caption = ParagraphStyle('caption', fontName='Times', fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=7)
    heading = ParagraphStyle('heading', fontName='TimesB', fontSize=12, leading=18, spaceBefore=6, spaceAfter=15)
    cell = ParagraphStyle('cell', fontName='Times', fontSize=9, leading=11, alignment=TA_CENTER)
    summary = pd.read_csv(SOURCE / 'resumo.csv')
    pair = pd.read_csv(DATA / 'comparacao_pareada_resumo.csv')
    forecasts = pd.read_csv(DATA / 'previsao_horizontes_2025.csv')
    costs = pd.read_csv(DATA / 'custos_resumo.csv')
    series = pd.read_csv(SOURCE / 'series.csv', parse_dates=['date'])
    stats = json.loads((DATA/'estatisticas.json').read_text(encoding='utf-8'))
    common = summary[summary.scenario=='mesmos_ativos']
    free = summary[summary.scenario=='escolha_livre']
    manifest = json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['bl_parameters']['NEWS_COUNT_MIN'] == 3, 'Capítulo desta rodada exige gate=3'
    def value(n, model='BL'):
        return free[(free.n==n)&(free.model==model)].iloc[0]
    def cost_value(n, model='BL'):
        return costs[(costs.scenario=='escolha_livre')&(costs.n==n)&(costs.model==model)].iloc[0]
    def common_value(n, model='BL'):
        return common[(common.n==n)&(common.model==model)].iloc[0]
    b20, m20 = value(20), value(20, 'Markowitz')
    best_return = free[free.model=='BL'].sort_values('return_median', ascending=False).iloc[0]
    best_sharpe = free[free.model=='BL'].sort_values('sharpe_median', ascending=False).iloc[0]
    news = stats['news']
    pages, blocks = [], []
    def page():
        nonlocal blocks
        if blocks:
            pages.append(blocks)
        blocks=[]
    def p(text): blocks.append(('p', text))
    def h(text, number): blocks.append(('h', (text,number)))
    def note(text): blocks.append(('note',text))
    def table(title, headers, rows, foot): blocks.append(('table',(title, headers, rows, foot)))
    def fig(title, drawing, name, foot): blocks.append(('figure',(title,drawing,name,foot)))

    # PAGINA 1
    h('ANÁLISE E DISCUSSÃO DE RESULTADOS','4')
    p('Este capítulo examina o desempenho das carteiras de Black-Litterman (BL) híbrido e de Markowitz em função do número de ativos, separando o efeito da alocação do efeito da seleção. A evidência principal corresponde ao experimento com cardinalidade n de 1 a 20, executado de 2 de janeiro a 30 de dezembro de 2025. A referência livre de risco é uma capitalização hipotética de 15% ao ano, e 2024 é a janela anterior utilizada para escolher os horizontes quantitativos.')
    h('Delimitação do experimento e critérios de comparação','4.1')
    p('No cenário de ativos comuns, cada subconjunto permanece fixo no ano e é recebido pelos dois modelos, que atualizam os pesos mensalmente. No cenário de seleção própria, cada estratégia escolhe suas n ações a cada rebalanceamento. O primeiro desenho compara métodos de alocação sob as mesmas oportunidades; o segundo compara processos completos de seleção e alocação.')
    table('Condições da rodada consolidada.', ['Elemento','Condição'], [
        ['Capital inicial e período','R$ 100.000; 250 observações em 2025'],
        ['Universo e cardinalidade','20 ações; n = 1, 2, ..., 20'],
        ['Cenário comum','551 subconjuntos; até 30 por n; semente fixa'],
        ['Seleção própria','20 trajetórias por modelo; seleção mensal'],
        ['Total','1.142 simulações individuais'],
        ['Estimação e risco','Janela de 126 pregões; aversão ao risco = 3'],
        ['Posições e limites','Compra; piso 0,1%; teto max(34,1677%; 1/n)'],
        ['Execução e custos','Fechamento seguinte ao sinal; 10 bps de turnover'],
        ['Prior e notícias','Equilíbrio equiponderado; gate de 3 manchetes']
    ], 'Fonte: elaboração própria, a partir do manifesto da simulação.')
    p('São utilizadas todas as 20 combinações possíveis em n = 1 e n = 19 e o único conjunto em n = 20. Entre n = 2 e n = 18, avaliam-se 30 subconjuntos distintos por tamanho, sem enumeração exaustiva. As medianas e os percentis descrevem a dispersão entre conjuntos; não são intervalos de confiança nem a trajetória de uma carteira específica.')
    note('O gate de três notícias foi adotado após comparar os limiares de 1 a 5 em 2025. Esta rodada é exploratória, não um teste independente do limiar. Os snapshots antigos não são agregados a estes resultados; a validade é discutida na Seção 4.7.')
    page()

    # PAGINA 2
    h('Desempenho quantitativo e escolha dos horizontes','4.2')
    p('Os arquivos de previsões foram produzidos pelo procedimento walk-forward e reutilizados nesta avaliação. O horizonte de cada ativo foi escolhido pela acurácia direcional na validação de 2024, aceitando apenas alvos integralmente conhecidos antes de 2025. A Tabela 4.2 confronta essa medida com o diagnóstico de 2025, limitado a previsões cujo desfecho termina até 30 de dezembro. O número de pares é 245, 240 ou 229 para horizontes de 5, 10 ou 21 pregões, respectivamente.')
    table('Desempenho das previsões utilizadas no experimento.',
          ['Ativo','h','HR val. (%)','HR 2025 (%)','Maior classe (%)','R² 2025'],
          [[r.ticker,str(r.horizon),fmt(r.validation_hr),fmt(r.test_hr),fmt(r.majority_hr),fmt(r.r2_test,3)] for r in forecasts.itertuples()],
          'Fonte: elaboração própria. HR: acurácia direcional; h: pregões. A maior classe é uma referência descritiva ex post; não é uma regra de negociação escolhida previamente.')
    p('A média simples da acurácia entre ativos cai de 60,29% na validação para 52,44% em 2025, redução de 7,85 pontos percentuais. O contraste evidencia que a seleção de um horizonte favorável em uma janela não assegura persistência no período seguinte. A avaliação financeira exige, portanto, análise própria, sem transformar a acurácia em expectativa direta de retorno.')
    page()

    # PAGINA 3
    h('Horizontes vencedores e alcance do sinal preditivo',None)
    table('Distribuição dos horizontes e diagnóstico agregado.', ['Indicador','Resultado'],[
        ['Horizonte de 5 pregões','3 ativos (15%)'],['Horizonte de 10 pregões','5 ativos (25%)'],
        ['Horizonte de 21 pregões','12 ativos (60%)'],['Mudanças frente ao vencedor do OOF completo','4 dos 20 ativos'],
        ['Acurácia média em 2025','52,44%'],['R² negativo em 2025','19 dos 20 ativos'],
        ['HR acima da classe majoritária','4 ativos; 1 empate; 15 abaixo']
    ], 'Fonte: elaboração própria, com base nas previsões por horizonte e nos preços locais.')
    p('A predominância do horizonte de 21 pregões indica que a seleção temporal favoreceu sinais mais longos para a maior parte do universo, sem eliminar heterogeneidade entre empresas. A nova regra altera os horizontes de BBDC4, EQTL3, PETR3 e SUZB3 em relação aos arquivos de vencedores do OOF completo. Não se deve, assim, atribuir à rodada atual as métricas dos vencedores históricos sem verificar qual previsão foi efetivamente utilizada.')
    p('No conjunto completo dos vencedores históricos, a acurácia direcional média armazenada é 55,78%, com R² negativo em 18 dos 20 ativos. Esses números pertencem a uma agregação mais ampla e a escolhas de estimador que podem consultar períodos posteriores a 2024. Por essa razão, funcionam como diagnóstico do treinamento, enquanto a Tabela 4.2 é a referência quantitativa desta discussão de 2025.')
    p('Os extremos do teste são SBSP3, com 70,31% de acerto direcional, e VALE3, com 33,75%. Entretanto, SBSP3 apenas empata com a frequência da classe majoritária, também de 70,31%. Acertar mais da metade das observações não implica superar um referencial trivial de sinal. PETR4, RADL3, RAIL3 e WEGE3 superam essa referência descritiva; a maioria dos ativos permanece abaixo dela.')
    p('O R² avalia a magnitude prevista e pode ser negativo mesmo com acurácia direcional superior a 50%. Em 2025, WEGE3 é a única exceção positiva, com R² de 0,153. O resultado dos demais ativos sugere erro quadrático superior ao da média retrospectiva da própria amostra avaliada, sem que essa média constitua uma previsão negociável. Como os alvos de vários pregões se sobrepõem, as observações não são independentes: não se interpreta a diferença para 50% como significância estatística automática.')
    p('A conclusão quantitativa é de sinal heterogêneo e fraco na magnitude. Uma eventual vantagem da carteira precisa ser relacionada à combinação de prior, covariância, restrições e custos, e não tomada como prova isolada de qualidade do XGBoost.')
    page()

    # PAGINA 4
    h('Comparação com ativos comuns','4.3')
    p('A comparação pareada controla a composição elegível: os dois modelos recebem os mesmos papéis e operam nas mesmas datas. A diferença mediana é calculada primeiro dentro de cada par e depois agregada; ela pode diferir da subtração das duas medianas marginais. A Selic acumula 14,87% nas 250 datas, em vez de 15%, devido à convenção de 252 dias úteis por ano.')
    table('Retornos líquidos nos subconjuntos comuns.',
          ['n','Conjuntos','BL med. (%)','MV med. (%)','Δ pareado (p.p.)','BL > MV (%)'],
          [[str(r.n),str(r.repetitions),fmt(r.return_bl),fmt(r.return_mk),fmt(r.paired_delta_median),fmt(r.win_pct,1)] for r in pair.itertuples()],
          'Fonte: elaboração própria. MV: Markowitz. A última coluna contabiliza vitórias estritas; os casos n = 1 e n = 2 têm 100% de empates.')
    p(f'A vantagem varia com a cardinalidade e a composição. Para n = 5, BL supera Markowitz em {fmt(pair[pair.n==5].win_pct.iloc[0],1)}% dos conjuntos, com diferença pareada mediana de {fmt(pair[pair.n==5].paired_delta_median.iloc[0])} p.p. Para n = 10, a frequência é {fmt(pair[pair.n==10].win_pct.iloc[0],1)}% e a diferença mediana é {fmt(pair[pair.n==10].paired_delta_median.iloc[0])} p.p. Essas frequências descrevem os subconjuntos avaliados, não probabilidades de vitória em outro ano.')
    page()

    # PAGINA 5
    h('Dispersão entre conjuntos e risco das carteiras',None)
    lines=[]
    for model,color in [('BL',BLUE),('Markowitz',ORANGE)]:
        g=common[common.model==model].sort_values('n')
        lines.append((model,g.n,g.return_median,color,g.return_p10,g.return_p90))
    lines.append(('Selic',[1,20],[14.8725102173]*2,colors.grey,None,None))
    drawing=line_chart('ativos_comuns',[('Ativos comuns: retorno líquido (%)',lines)],bands=True)
    fig('Retorno por cardinalidade com controle dos ativos elegíveis.',drawing,'ativos_comuns',
        'Fonte: elaboração própria. Faixas: percentis entre subconjuntos, sem interpretação de intervalo de confiança.')
    rows=[]
    for n in [1,5,10,20]:
        for model in ['BL','Markowitz']:
            r=common[(common.n==n)&(common.model==model)].iloc[0]
            rows.append([str(n),model,fmt(r.sharpe_median),fmt(r.volatility_median),fmt(r.drawdown_median)])
    table('Risco e desempenho ajustado nos conjuntos comuns.', ['n','Modelo','Sharpe med.','Vol. anual (%)','MDD med. (%)'],rows,
          'Fonte: elaboração própria. Cada coluna é uma mediana marginal; não representa necessariamente o mesmo subconjunto.')
    p(f'Para BL, os percentis de retorno em n = 5 são {fmt(common_value(5).return_p10)}% e {fmt(common_value(5).return_p90)}%; em n = 15, são {fmt(common_value(15).return_p10)}% e {fmt(common_value(15).return_p90)}%. A amplitude entre esses valores descreve a dependência da composição sorteada. Sua comparação permite avaliar robustez dentro do universo, sem assegurar generalização temporal.')
    p('Em n = 1, os métodos coincidem por investirem integralmente na mesma ação. Em n = 2, o teto de 50% impõe pesos-alvo iguais. Para n = 3, o teto de 34,17% deixa pouca liberdade de alocação. Portanto, as diferenças nos primeiros pontos também refletem restrições geométricas, e não apenas qualidade das estimativas.')
    page()

    # PAGINA 6
    h('Comparação com seleção própria dos ativos','4.4')
    p('Na escolha livre, cada modelo ranqueia os ativos pelo retorno esperado excedente dividido pela volatilidade histórica. Markowitz utiliza a média móvel; BL utiliza o posterior. A regra é explicitamente acrescentada à alocação: não corresponde à solução exaustiva do problema combinatório de selecionar a melhor carteira com n ações. Os retornos abaixo incluem seleção e distribuição de pesos.')
    rows=[]
    for n in range(1,21):
        b=free[(free.n==n)&(free.model=='BL')].iloc[0]
        m=free[(free.n==n)&(free.model=='Markowitz')].iloc[0]
        rows.append([str(n),fmt(b.return_median),fmt(m.return_median),fmt(b.sharpe_median),fmt(m.sharpe_median),fmt(b.drawdown_median),fmt(m.drawdown_median)])
    table('Desempenho das estratégias com seleção própria.',
          ['n','Ret. BL (%)','Ret. MV (%)','Sharpe BL','Sharpe MV','MDD BL (%)','MDD MV (%)'],rows,
          'Fonte: elaboração própria. Selic no período: 14,87%. Há uma trajetória dinâmica por modelo e cardinalidade.')
    wins = int((free.pivot(index='n', columns='model', values='return_median').BL > free.pivot(index='n', columns='model', values='return_median').Markowitz).sum())
    p(f'BL supera Markowitz em retorno para {wins} das 20 cardinalidades. Com uma ação, seu retorno é {fmt(value(1).return_median)}%, contra {fmt(value(1,"Markowitz").return_median)}% de Markowitz, e seu drawdown é {fmt(value(1).drawdown_median)}%. Nesse caso, a distribuição de pesos é trivial: toda diferença decorre das ações escolhidas e das operações. A concentração exige atenção própria, mesmo quando carteiras maiores apresentam resultados favoráveis.')
    page()

    # PAGINA 7
    h('Trajetórias de capital e comparação com a renda fixa',None)
    panels=[]
    for n in [5,20]:
        lines=[]
        for model,color in [('BL',BLUE),('Markowitz',ORANGE)]:
            g=series[(series.scenario=='escolha_livre')&(series.n==n)&(series.model==model)].sort_values('date')
            lines.append((model,np.arange(len(g)),g.capital,color,None,None))
        benchmark=100000*1.15**(np.arange(1,251)/252)
        lines.append(('Selic',np.arange(250),benchmark,colors.grey,None,None))
        panels.append((f'Seleção própria: n = {n}',lines))
    drawing=line_chart('trajetorias_capital',panels,capital=True)
    fig('Capital realizado em duas cardinalidades ilustrativas.',drawing,'trajetorias_capital',
        'Fonte: elaboração própria. Capital inicial de R$ 100 mil; custos de 10 bps. n = 5 é um exemplo, não uma seleção de tamanho feita antes do teste.')
    table('Resultado com todo o universo de vinte ativos.', ['Indicador','BL','Markowitz','Selic'],[
        ['Capital final (R$)',fmt(100000*(1+b20.return_median/100)),fmt(100000*(1+m20.return_median/100)),'114.872,51'],
        ['Retorno líquido (%)',fmt(b20.return_median),fmt(m20.return_median),'14,87'],
        ['Volatilidade anual (%)',fmt(b20.volatility_median),fmt(m20.volatility_median),'0,00'],
        ['Sharpe sobre a Selic',fmt(b20.sharpe_median),fmt(m20.sharpe_median),'Não definido'],
        ['Drawdown máximo (%)',fmt(b20.drawdown_median),fmt(m20.drawdown_median),'0,00']
    ], 'Fonte: elaboração própria. A Selic é uma série determinística; seu Sharpe não é definido por ter excesso e volatilidade nulos.')
    p(f'Em n = 20, os cenários coincidem, pois todos os ativos são elegíveis. A diferença de retorno BL menos Markowitz é {fmt(b20.return_median-m20.return_median)} p.p.; frente à Selic, é {fmt(b20.return_median-b20.selic_return_pct)} p.p. As volatilidades são {fmt(b20.volatility_median)}% e {fmt(m20.volatility_median)}%, respectivamente. Essa comparação deve considerar também os drawdowns, pois a renda fixa não apresenta as oscilações das ações neste modelo simplificado.')
    p(f'O maior retorno de BL na escolha livre ocorre em n = {int(best_return.n)}, com {fmt(best_return.return_median)}%; seu maior Sharpe ocorre em n = {int(best_sharpe.n)}, com {fmt(best_sharpe.sharpe_median)}. Esses máximos foram identificados retrospectivamente, não validam um tamanho ótimo antes do teste. A capitalização estimada dos antigos CSVs não é comparada a estas curvas, pois pertence a outro motor e não constitui previsão compatível com este experimento.')
    page()

    # PAGINA 8
    h('Sensibilidade à cardinalidade e aos custos de transação','4.5')
    p(f'Na escolha livre de BL, os retornos em n = 1, n = 2 e n = 20 são, respectivamente, {fmt(value(1).return_median)}%, {fmt(value(2).return_median)}% e {fmt(b20.return_median)}%. A volatilidade passa de {fmt(value(1).volatility_median)}% para {fmt(b20.volatility_median)}% entre os extremos. Não se presume melhora monotônica: aumentar n modifica as ações selecionadas, o risco de concentração e a liberdade de otimização.')
    rows=[]
    for n in [1,2,5,10,20]:
        for model in ['BL','Markowitz']:
            r=costs[(costs.scenario=='escolha_livre')&(costs.n==n)&(costs.model==model)].iloc[0]
            rows.append([str(n),model,fmt(r.gross_return),fmt(r.net_return),fmt(r.impact_pp),fmt(r.explicit_cost_brl)])
    table('Efeito dos custos nas estratégias com seleção própria.',
          ['n','Modelo','Sem custo (%)','Líquido (%)','Impacto (p.p.)','Custo pago (R$)'],rows,
          'Fonte: elaboração própria. O contrafactual sem custo mantém os mesmos pesos e operações, removendo apenas o desconto financeiro; não há nova otimização.')
    p(f'O custo nominal pago e a perda de capital final não são a mesma medida: uma despesa antecipada também reduz o capital sujeito a retornos posteriores. Para n = 20, BL paga R$ {fmt(cost_value(20).explicit_cost_brl)}, com impacto de {fmt(cost_value(20).impact_pp)} p.p. de retorno. Markowitz paga R$ {fmt(cost_value(20,"Markowitz").explicit_cost_brl)}, com impacto de {fmt(cost_value(20,"Markowitz").impact_pp)} p.p.; seu turnover é {fmt(cost_value(20,"Markowitz").turnover)}, contra {fmt(cost_value(20).turnover)} de BL.')
    p(f'Em n = 2, BL tem turnover de {fmt(cost_value(2).turnover)} e custo de R$ {fmt(cost_value(2).explicit_cost_brl)}, com impacto de {fmt(cost_value(2).impact_pp)} p.p. Embora os pesos-alvo sejam 50/50, a troca das ações escolhidas e a restauração das proporções geram negociação. A comparação entre retorno bruto e líquido permite distinguir desempenho financeiro e despesas de implementação.')
    p('A rodada atual fixa a frequência mensal. Os antigos resultados diários e semanais e os experimentos de remoção de ativos utilizam outro motor e não são incorporados como sensibilidade diretamente comparável. A dispersão entre subconjuntos é uma análise de composição, mas não equivale a uma exclusão controlada de um ativo específico. Essas verificações exigem nova execução homogênea antes de receberem conclusões numéricas.')
    page()

    # PAGINA 9
    h('Cobertura de notícias e influência efetiva das views','4.6')
    p('O experimento utiliza gate de três notícias: a presença de pelo menos três manchetes no dia elegível habilita a view híbrida. Abaixo desse limiar, o hard prior reduz sua confiança a um valor próximo de zero. Para avaliar a influência potencial das views, a auditoria deve incidir sobre as datas mensais de decisão, e não apenas sobre a cobertura diária do ano.')
    table('Cobertura nas datas de formação dos sinais mensais.', ['Indicador','Quantidade','Percentual'],[
        ['Pares ativo-data avaliados','240','100,00%'],
        ['Com alguma notícia no dia elegível','40','16,67%'],
        ['Com ao menos três notícias (gate = 3)',str(news['eligible']),fmt(news['eligible']/news['observations']*100)+'%'],
        ['Abaixo do gate',str(news['observations']-news['eligible']),fmt((1-news['eligible']/news['observations'])*100)+'%'],
        ['Datas mensais com alguma view elegível',str(news['dates_with_any_eligible'])+' de 12',fmt(news['dates_with_any_eligible']/12*100)+'%']
    ], 'Fonte: elaboração própria, com base na auditoria de sinais. Notícias elegíveis: dia civil anterior ao sinal; universo de 20 ativos.')
    p(f'O gate habilita {news["eligible"]} dos {news["observations"]} pares ativo-data, presentes em {news["dates_with_any_eligible"]} das doze datas mensais. Nas demais decisões, nenhuma ação atinge o mínimo de três manchetes. Esses números expressam cobertura informacional, não o tamanho exato da contribuição das views aos pesos. A regra é mais restritiva que a mera presença de uma notícia, embora quarenta pares contenham alguma manchete.')
    p('Nos ativos sem notícias elegíveis, a atualização permanece próxima do prior; nos demais, a mistura quantitativa e textual participa conforme a confiança estimada. A influência de uma opinião pode alcançar outros papéis pela estrutura de covariância. Por isso, o percentual de pares habilitados não deve ser interpretado diretamente como percentual da rentabilidade explicado pelas views.')
    p('A consulta apenas ao dia civil anterior ao sinal e a frequência mensal também limitam a informação utilizada: notícias em outros dias não geram rebalanceamento adicional. O gate de três foi adotado após a sensibilidade de 1 a 5 no próprio ano de 2025. Assim, o resultado descreve uma escolha retrospectiva, sem constituir validação independente do limiar nem evidência de irrelevância do sinal textual.')
    p('Não se compararam BL sem views, BL apenas quantitativo, BL apenas textual e híbrido completo sob um protocolo independente. Essa ablação continua necessária para atribuir ganhos a cada componente. A comparação atual identifica o desempenho do processo combinado frente a Markowitz e à Selic, mas não isola causalmente o valor incremental do XGBoost ou do sentimento.')
    page()

    # PAGINA 10
    h('Limitações e alcance das conclusões','4.7')
    p('A primeira limitação é temporal. Os horizontes foram escolhidos em 2024, mas a seleção original entre classificador e regressor pode ter consultado todo o OOF. Não houve retreinamento com seleção aninhada anterior ao teste. Além disso, o gate de três foi escolhido após observar o desempenho de 2025. A causalidade corrigida no motor não elimina esses problemas de seleção. A rodada é exploratória, não uma comprovação definitiva fora da amostra.')
    p('A segunda limitação é estatística. Há apenas um ano de avaliação financeira e vários tamanhos de carteira examinados. Os subconjuntos compartilham ações, datas e choques de mercado, de modo que 1.142 simulações não representam 1.142 amostras independentes. As frequências de vitória são descritivas. A escolha do melhor n pelo retorno de 2025 favoreceria uma configuração ex post; sua utilidade operacional exige validação anterior e teste em outro período, além de procedimentos que respeitem a dependência temporal.')
    p(f'A terceira limitação é de atribuição. Com gate de três notícias, {news["eligible"]} dos {news["observations"]} pares ativo-data habilitam as views. A comparação não separa os efeitos do prior, das restrições, da covariância e das fontes de inteligência computacional. O diagnóstico quantitativo, com R² negativo em 19 ativos e acurácia inferior à classe majoritária em 15, reforça a necessidade de evitar inferências causais apenas pelo retorno agregado.')
    p('A quarta limitação é operacional. O universo é fixo, está sujeito a viés de sobrevivência e inclui classes de ações da mesma empresa, como PETR3 e PETR4; vinte papéis não equivalem a vinte fontes independentes de risco. A execução ao fechamento seguinte é uma aproximação, os custos são lineares e não se simulam spread, impacto de mercado, tributação ou liquidação terminal. O piso de 0,1% mantém exatamente n posições, mas algumas podem ser economicamente pequenas.')
    p('Por fim, o prior neutro evita usar capitalização observada posteriormente, mas se afasta da especificação de equilíbrio ponderado por valor de mercado apresentada na arquitetura. A escala diária das views também é aproximada. O referencial de 15% ao ano é uma taxa constante hipotética, e não a trajetória histórica da Selic. Essas convenções delimitam a comparação e precisam permanecer explícitas no procedimento experimental.')
    p(f'No universo completo, BL retorna {fmt(b20.return_median)}%, frente aos {fmt(m20.return_median)}% de Markowitz e aos {fmt(b20.selic_return_pct)}% da Selic. A vantagem nesta rodada com gate de três não demonstra superioridade geral: o limiar foi escolhido com conhecimento de 2025. A escolha de n e a contribuição de cada fonte também exigem validação independente.')
    pages.append(blocks)

    story, tex, table_no, fig_no = [], [], 0, 0
    for pi, page_blocks in enumerate(pages):
        if pi:
            story.append(PageBreak())
            tex.append('\\clearpage\n')
        for kind, content in page_blocks:
            if kind=='h':
                title, number=content
                rendered = f'{number}  {title}' if number else title
                story.append(Paragraph(escape(rendered), heading))
                if number=='4':
                    tex.append('\\chapter{'+latex(title)+'}\n\\label{cap:resultados}\n'
                               '\\renewcommand{\\thetable}{\\thechapter.\\arabic{table}}\n'
                               '\\renewcommand{\\thefigure}{\\thechapter.\\arabic{figure}}\n'
                               '\\setcounter{table}{0}\\setcounter{figure}{0}\n')
                elif number: tex.append('\\section{'+latex(title)+'}\n')
                else: tex.append('\\subsection*{'+latex(title)+'}\n')
            elif kind in ('p','note'):
                story.append(Paragraph(escape(content), body if kind=='p' else small))
                tex.append(('\\noindent{\\footnotesize ' if kind=='note' else '')+latex(content)+('}' if kind=='note' else '')+'\n\n')
            elif kind=='table':
                title, headers, rows, foot=content
                table_no+=1
                story.append(Paragraph(f'Tabela 4.{table_no} - '+escape(title),caption))
                if len(headers)==2:
                    widths=[WIDTH*.43,WIDTH*.57]
                else:
                    widths=[WIDTH/len(headers)]*len(headers)
                cells=[[Paragraph(escape(str(c)),cell) for c in row] for row in [headers]+rows]
                t=Table(cells,colWidths=widths,hAlign='CENTER',repeatRows=1)
                t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.6,colors.black),
                    ('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),
                    ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
                    ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
                    ('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
                story.extend([t,Spacer(1,5),Paragraph(escape(foot),small)])
                n=len(headers)
                cols=('p{0.40\\linewidth}p{0.52\\linewidth}' if n==2 else '@{}'+'r'*n+'@{}')
                tex.append('\\begin{table}[H]\n\\centering\n\\caption{'+latex(title)+'}\n{\\footnotesize\n\\begin{tabular}{'+cols+'}\n\\hline\n')
                tex.append(' & '.join(latex(str(c)) for c in headers)+' \\\\ \\hline\n')
                for row in rows: tex.append(' & '.join(latex(str(c)) for c in row)+' \\\\\n')
                tex.append('\\hline\n\\end{tabular}}\n\\par\\smallskip{\\footnotesize '+latex(foot)+'}\n\\end{table}\n')
            elif kind=='figure':
                title,drawing,name,foot=content
                fig_no+=1
                story.extend([Paragraph(f'Figura 4.{fig_no} - '+escape(title),caption),drawing,Paragraph(escape(foot),small)])
                tex.append('\\begin{figure}[H]\n\\centering\n\\caption{'+latex(title)+'}\n\\includegraphics[width=\\textwidth]{05_Resultados/figuras/'+name+'.pdf}\n\\par\\smallskip{\\footnotesize '+latex(foot)+'}\n\\end{figure}\n')
    def header(canvas, doc):
        canvas.setFont('Times',10)
        canvas.drawRightString(A4[0]-2*cm,A4[1]-1.8*cm,str(doc.page))
    document=SimpleDocTemplate(str(PDFOUT/'capitulo_4_resultados.pdf'),pagesize=A4,
                              leftMargin=3*cm,rightMargin=2*cm,topMargin=2.6*cm,bottomMargin=2*cm,
                              title='Capítulo 4 - Análise e discussão de resultados',author='Igor Rabelo de Toledo')
    document.build(story,onFirstPage=header,onLaterPages=header)
    (TEXOUT/'resultados.tex').write_text(''.join(tex),encoding='utf-8')
    shutil.copy2(PDFOUT/'capitulo_4_resultados.pdf',TEXOUT/'capitulo_4_resultados.pdf')
    metadata=dict(pages_planned=len(pages),tables=table_no,figures=fig_no,news_gate=3,
                  numeric_sources=stats['source_hashes'],
                  context_pdf=r'C:\Users\igor.toledo_visagio\Downloads\Trabalho_de_Conclusão_de_Curso___Igor_Toledo (5).pdf')
    (WORK/'manifesto_capitulo.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
