"""Confere a versao do TCC que apresenta somente o experimento pool."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from collections import Counter
import difflib
import hashlib
import json
import re
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
TCC=Path(r'C:/Users/igor.toledo_visagio/Documents/TCC 2')
OUT=sorted((REPO/'output').glob('tcc_versao_pool_*'))[-1]


def replace(path,old,new):
    raw=path.read_bytes();text=raw.decode('utf-8').replace('\r\n','\n')
    assert text.count(old)==1,old[:100]
    path.write_bytes(text.replace(old,new,1).replace('\n','\r\n' if b'\r\n' in raw else '\n').encode('utf-8'))


def align_architecture():
    p=TCC/'03_Arquitetura/arquitetura.tex'
    if 'O horizonte operacional do sistema é diário no que diz respeito' not in p.read_text(encoding='utf-8'):return
    replace(p,
        'O horizonte operacional do sistema é diário no que diz respeito à formação do vetor posterior $\\boldsymbol{\\mu}$ e dos pesos da carteira. As opiniões quantitativas, entretanto, podem ser geradas em horizontes múltiplos de dias úteis. Após sua geração, esses sinais são convertidos para uma escala diária, permitindo que sejam combinados com o sinal textual e utilizados de forma consistente pelo modelo de alocação.',
        'Os retornos e as opiniões são expressos em escala diária, mas a formação dos sinais e o rebalanceamento da carteira ocorrem mensalmente. As opiniões quantitativas podem ser geradas em horizontes múltiplos de dias úteis. Após sua geração, esses sinais são convertidos para a escala diária, permitindo que sejam combinados com o sinal textual e utilizados de forma consistente pelo modelo de alocação.')
    replace(p,
        'Após a obtenção do vetor posterior $\\boldsymbol{\\mu}_t$, os pesos preliminares são calculados pela solução do problema de média-variância com restrição de orçamento integral, ainda sem impor as restrições de não negatividade e de concentração máxima.',
        'Após a obtenção do vetor posterior $\\boldsymbol{\\mu}_t$, a carteira é calculada pelo problema de média-variância com orçamento integral e limites por ativo. A solução sem esses limites, apresentada a seguir, explicita a relação entre retorno esperado, covariância e aversão ao risco; a implementação resolve diretamente o problema restrito por gradiente projetado com aceleração.')
    replace(p,
        'A solução preliminar pode conter posições negativas, correspondentes a vendas a descoberto, e concentrações superiores aos limites definidos para a carteira. Por essa razão, os pesos $\\boldsymbol{w}^{\\mathrm{MV}}_t$ são submetidos a uma etapa posterior de projeção sobre o conjunto factível.',
        'A solução sem limites pode conter posições negativas e concentrações superiores aos limites definidos para a carteira. Na implementação, as restrições são impostas em cada iteração da otimização. A projeção é parte desse processo iterativo, em vez de um ajuste único da solução sem limites.')
    replace(p,'w_j\\geq0,','w_j\\geq0{,}001,')
    replace(p,
        'em que $w_{\\max}$ representa a concentração máxima permitida para cada ativo. Em termos práticos, as três condições de $\\mathcal{W}$ estabelecem que não é permitido operar vendido, ou seja, assumir pesos negativos; que nenhum ativo pode concentrar mais capital do que o teto preestabelecido; e que todo o capital disponível deve ser alocado, para que a soma dos pesos resulte sempre em $1$.',
        'em que $w_{\\max}=\\max(0{,}341677,1/n)$ representa a concentração máxima permitida para cada ativo. As condições são aplicadas aos n ativos selecionados; os demais recebem peso zero. Não é permitido operar vendido, cada posição selecionada tem piso de 0,1\\%, nenhum ativo pode ultrapassar o teto e todo o capital disponível deve ser alocado, para que a soma dos pesos resulte em $1$.')
    replace(p,
        'Como o vetor de pesos preliminares ($\\boldsymbol{w}^{\\mathrm{MV}}_t$) gerado pelo modelo Black-Litterman frequentemente viola essas regras, sugerindo alocações extremas, é necessário ajustá-lo. A carteira final é obtida por meio de uma projeção euclidiana, cujo objetivo é encontrar a alocação estritamente válida que seja o mais fiel possível às proporções sugeridas pelo modelo original:',
        'Em cada iteração, o método avança na direção de melhora do objetivo de média-variância e projeta o vetor resultante no conjunto factível. A etapa de projeção pode ser escrita como:')
    replace(p,
        '    \\boldsymbol{w}^{\\star}_t\n    =\n    \\underset{\n        \\boldsymbol{w}\\in\\mathcal{W}\n    }{\\operatorname{argmin}}\n    \\;\n    \\left\\|\n        \\boldsymbol{w}\n        -\n        \\boldsymbol{w}^{\\mathrm{MV}}_t\n    \\right\\|_2^2.',
        '    \\boldsymbol{w}^{(k+1)}_t\n    =\n    \\underset{\\boldsymbol{w}\\in\\mathcal{W}}{\\operatorname{argmin}}\n    \\left\\|\\boldsymbol{w}-\\left[\\boldsymbol{v}^{(k)}_t\n    +\\eta\\left(\\boldsymbol{\\mu}_t-\\delta_t\\boldsymbol{\\Sigma}_t\\boldsymbol{v}^{(k)}_t\\right)\\right]\\right\\|_2^2,')
    replace(p,
        'Para que esse problema matemático tenha solução teórica e prática, impõe-se a restrição de factibilidade:',
        'em que $\\boldsymbol{v}^{(k)}_t$ é o iterante acelerado e $\\eta$ é o passo definido pela curvatura do objetivo. A convergência é verificada antes de aceitar os pesos. Para que o conjunto de restrições seja factível, impõe-se:')
    replace(p,
        'O processo ilustrado baseia-se em uma lógica de esgotamento. Na primeira iteração, o sistema analisa as recomendações preliminares. Se o modelo original sugerir uma posição negativa, que representa uma venda a descoberto para um ativo, seu peso real é imediatamente travado em zero. Da mesma forma, se a sugestão ultrapassar o limite máximo de segurança, a alocação é travada em $w_{\\max}$. ',
        'O processo ilustrado representa o ajuste aos limites em uma etapa de projeção. Os pesos são limitados ao piso das posições selecionadas e ao teto $w_{\\max}$, preservando o orçamento integral. ')
    replace(p,
        'O ciclo de correção se repete de forma automática até que nenhum ativo viole as regras estabelecidas. Ao término das iterações, obtém-se o vetor $\\boldsymbol{w}^{\\star}_t$, que representa a carteira definitiva perfeitamente ajustada à realidade operacional e pronta para execução.',
        'As etapas de avanço e projeção se repetem até a convergência do objetivo restrito. Ao término das iterações, obtém-se o vetor $\\boldsymbol{w}^{\\star}_t$, sujeito aos limites de posição e orçamento, para execução no fechamento seguinte ao sinal.')


def structural(text):
    clean=re.sub(r'(?<!\\)%[^\n]*','',text)
    stack=[]
    for kind,env in re.findall(r'\\(begin|end)\{([^}]+)\}',clean):
        if kind=='begin':stack.append(env)
        else:assert stack and stack.pop()==env,(env,stack)
    assert not stack,stack
    plain=re.sub(r'\\[^a-zA-Z]','',clean);depth=0
    for ch in plain:
        if ch=='{':depth+=1
        elif ch=='}':
            depth-=1
            assert depth>=0
    assert depth==0
    assert plain.count('$')%2==0


def main():
    align_architecture()
    backup=ZipFile(OUT/'backup_anterior.zip')
    current={p.relative_to(TCC).as_posix():p.read_text(encoding='utf-8') for p in TCC.rglob('*.tex')}
    old={n:backup.read(n).decode('utf-8').replace('\r\n','\n') for n in backup.namelist() if n.endswith('.tex')}
    alltext='\n'.join(current.values());oldtext='\n'.join(old.values())
    labels=Counter(re.findall(r'\\label\{([^}]+)\}',alltext))
    oldlabels=Counter(re.findall(r'\\label\{([^}]+)\}',oldtext))
    assert not [k for k,v in labels.items() if v>1 and v>oldlabels[k]]
    refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',alltext))
    oldrefs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',oldtext))
    assert not (refs-labels.keys())-(oldrefs-oldlabels.keys())
    for text in current.values():structural(text)
    results=current['05_Resultados/resultados.tex']
    for bad in ['BL original','BL atual','XGB original','Original &','rodada original','comparação complementar',
                'substituição do XGBoost','7,11','8,97','52,44','58,40','40,14','3.997']:
        assert bad not in results,bad
    for p in ['06_Conclusao/conclusao.tex','04_Procedimentos/procedimento.tex']:
        for bad in ['BL original','modelo original','testes complementares','comparação complementar','rodada complementar','7,11','8,97','52,44','3.997']:
            assert bad not in current[p],(p,bad)
    expected_models={'BL_pool','Markowitz','Prior','XGB_pool'}
    m=pd.read_csv(OUT/'metricas_utilizadas.csv')
    assert set(m.model)==expected_models and len(m)==2284
    source=pd.read_csv(REPO/'analise_resultados/simulacao_pool/metricas.csv')
    pd.testing.assert_frame_equal(m.reset_index(drop=True),source[source.model.isin(expected_models)].reset_index(drop=True))
    f=lambda v,n=2:f'{v:,.{n}f}'.replace(',','X').replace('.',',').replace('X','.')
    checked=0
    for n,g in m[m.scenario.eq('escolha_livre')].groupby('n'):
        g=g.set_index('model')
        cells=[str(n)]+[f(g.loc[a,c]) for c in ['return_pct','sharpe','max_drawdown_pct'] for a in ['BL_pool','Markowitz']]
        assert ' & '.join(cells) in results,(n,cells)
        checked+=1
    test=pd.read_csv(OUT/'previsoes_horizontes_utilizados.csv')
    for ticker,g in test.groupby('ticker'):
        g=g.set_index('phase');r=g.loc['test']
        cells=[ticker,str(int(r.horizon)),f(g.loc['validation','hr']),f(r.hr),f(r.majority),f(r.r2,3)]
        assert ' & '.join(cells) in results,(ticker,cells)
        checked+=1
    for r in pd.read_csv(OUT/'custos_utilizados.csv').itertuples():
        if r.model not in ['BL_pool','Markowitz'] or r.n not in [1,2,5,10,20]:continue
        cells=[str(r.n),'BL' if r.model=='BL_pool' else 'Markowitz']+[f(getattr(r,c)) for c in ['gross','net','impact','cost']]
        assert ' & '.join(cells) in results,cells
        checked+=1
    tables=re.findall(r'\\begin\{tabular\}\{(.*?)\}\n(.*?)\\end\{tabular\}',results,flags=re.S)
    assert len(tables)==11
    for spec,body in tables:
        cols=len(re.findall(r'p\{[^}]*\}|[lcr]',spec))
        for r in body.split('\\\\'):
            r=r.replace('\\hline','').strip()
            if r:assert r.count('&')+1==cols,(spec,r)
    images=[]
    for img in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',results):
        assert (TCC/img).is_file(),img
        assert (TCC/img).read_bytes()!=backup.read(img),img
        images.append(img)
    changed=[];diffs=[]
    for n,text in current.items():
        if text==old[n]:continue
        changed.append(n)
        diffs.extend(difflib.unified_diff(old[n].splitlines(keepends=True),text.splitlines(keepends=True),fromfile=n+' (antes)',tofile=n+' (depois)'))
    (OUT/'alteracoes.diff').write_text(''.join(diffs),encoding='utf-8')
    # Confere a reconstrucao de custos com uma simulacao independente, sem mudar artefatos.
    from intel.Blacklitterman.simular_dois_cenarios import simulate
    from intel.Blacklitterman.simular_pool_vs_atual import fixed_omega_posterior
    # O caso de pesos unicos permite testar custos sem depender de previsoes.
    dates=pd.date_range('2025-01-01',periods=5,freq='B')
    returns=pd.DataFrame({'A':[.01,-.02,.03,.01,.02]},index=dates)
    sig={dates[1]:dict(signal_date=dates[0],cov=np.array([[.001]]),TEST=np.array([.001]))}
    a,c,_=simulate(dates,returns,sig,1,'TEST',None,3.,10.)
    b,_,_=simulate(dates,returns,sig,1,'TEST',None,3.,0.)
    gross=(np.prod((c.capital+c.cost_brl)/c.capital.shift(1,fill_value=100000))-1)*100
    np.testing.assert_allclose(gross,b['return_pct'],atol=1e-9,rtol=0)
    verification=dict(status='passed',main_model='BL_pool',presented_models=sorted(expected_models),
        trajectories_presented=len(m),tables_checked=len(tables),numeric_rows_checked=checked,
        previous_version_comparison_removed=True,new_figures=images,cost_reconstruction_checked=True,
        structural_latex_checks=True,new_missing_references=False,compiled=False,
        compilation_note='Compilador LaTeX nao disponivel no ambiente.',modified_tex_files=changed)
    (OUT/'verificacao.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'LEIA-ME.txt').write_text(
        'TCC: somente a configuracao final de regressao XGBoost conjunta e BL com confianca de erros maduros.\n'
        'Os capitulos apresentam BL, Markowitz, prior sem views e XGBoost conjunto diretamente no otimizador.\n'
        'As comparacoes com a versao anterior foram retiradas. BL: 27,79% em vinte ativos, 43,72% em dez e 58,02% em cinco.\n'
        '2284 trajetorias utilizadas: 1142 principais e 1142 controles, extraidas dos resultados ja calculados.\n'
        'Graficos e tabelas foram refeitos a partir desses dados. O backup fica separado do projeto atualizado.\n'
        'Checagens estruturais e numericas passaram. PDF nao compilado: compilador LaTeX indisponivel.\n',encoding='utf-8')
    with ZipFile(OUT/'TCC_2_versao_final_pool.zip','w',ZIP_DEFLATED) as z:
        for p in TCC.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(TCC).as_posix())
    with ZipFile(OUT/'TCC_2_versao_final_pool.zip') as z:assert z.testzip() is None
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    print('Projeto:',OUT/'TCC_2_versao_final_pool.zip')


if __name__=='__main__':main()
