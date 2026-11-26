from pathlib import Path
import csv,json,hashlib,collections
root=Path('analise_resultados/camadas'); out=root/'revisao_300'
p=out/'anotacao_ia_cega.txt'
s=p.read_text(encoding='utf-8-sig').replace('3|P|','3|P?|').replace('19|N|','19|N?|').replace('228|N?|Retirada de referência reduz previsibilidade de preços','228|Z?|Mudança de política sem consequência direcional explícita').replace('251|N?|Prêmio elevado sugere risco de valuation','251|Z?|Valuation relativo sem recomendação explícita')
p.write_text(s,encoding='utf-8')
labels={'P':'positivo','N':'negativo','Z':'neutro'}
a={}
for line in s.splitlines():
    i,code,why=line.split('|',2)
    assert int(i) not in a
    a[int(i)]={'rotulo_ia':labels[code[0]],'revisao_recomendada':'sim' if '?' in code else 'nao','justificativa_ia':why}
assert set(a)==set(range(1,301))
# Os rótulos individuais anteriores são lidos somente após fechar a anotação acima.
with (root/'rotulacao_humana.csv').open(encoding='utf-8-sig',newline='') as f: human=list(csv.DictReader(f))
with (root/'gabarito_modelo.csv').open(encoding='utf-8-sig',newline='') as f: original={int(r['sample_id']):r['sentimento'] for r in csv.DictReader(f)}
assert len(human)==300 and {int(r['sample_id']) for r in human}==set(a)
rows=[]
for h in human:
    i=int(h['sample_id']); lab=h['rotulo_avaliador_1'].strip().lower()
    assert lab in labels.values(), (i,lab)
    ia=a[i]['rotulo_ia']
    rows.append({k:h[k] for k in ('sample_id','ticker','titulo','data')}|{'rotulo_humano_original':lab,**a[i],'rotulo_modelo_original':original[i],'situacao':'acordo_provisorio' if lab==ia else 'divergencia_pendente','rotulo_acordo_provisorio':lab if lab==ia else '', 'consenso_confirmado':'','observacao_revisao':''})
def save(name,rs,fields=None):
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rs[0]));w.writeheader();w.writerows(rs)
save('revisao_consenso.csv',rows)
save('divergencias_humano_ia.csv',[r for r in rows if r['situacao']=='divergencia_pendente'])
save('avaliacao_ia_independente.csv',[{k:r[k] for k in ('sample_id','ticker','titulo','data','rotulo_ia','revisao_recomendada','justificativa_ia')} for r in rows])
classes=['positivo','negativo','neutro']
def comparison(x,y):
    n=len(rows); conf={k:{v:sum(r[x]==k and r[y]==v for r in rows) for v in classes} for k in classes}
    observed=sum(r[x]==r[y] for r in rows)/n
    expected=sum(sum(r[x]==k for r in rows)*sum(r[y]==k for r in rows) for k in classes)/n**2
    return {'n':n,'acordos':round(observed*n),'concordancia':observed,'kappa':(observed-expected)/(1-expected),'matriz_linhas_primeiro_colunas_segundo':conf}
report={'n':len(rows),'tipo_segundo_avaliador':'IA; nao e segundo avaliador humano','modelo_original':'gpt-5.6-luna','avaliador_atual':'Codex nesta conversa; classificacao contextual, sem nova chamada ao modelo original','cegamento':'Rotulos individuais humanos e originais nao foram exibidos antes da anotacao. O historico desta conversa ja continha metricas agregadas e a suspeita de vies; nao constitui cegamento experimental completo.','escopo':'Somente titulo e ticker. Sem consulta a artigos, precos futuros ou fontes externas.','rotulos_frequencias':{c:dict(collections.Counter(r[c] for r in rows)) for c in ['rotulo_humano_original','rotulo_ia','rotulo_modelo_original']},'humano_ia':comparison('rotulo_humano_original','rotulo_ia'),'humano_modelo_original':comparison('rotulo_humano_original','rotulo_modelo_original'),'ia_modelo_original':comparison('rotulo_ia','rotulo_modelo_original'),'revisao_recomendada_ia':sum(r['revisao_recomendada']=='sim' for r in rows),'consenso_final':'Pendente de adjudicacao pelo usuario; campos confirmados vazios. Acordo provisório nao e consenso confirmado.','sha256_anotacao_antes_comparacao':hashlib.sha256(p.read_bytes()).hexdigest(),'sha256_fonte_humana':hashlib.sha256((root/'rotulacao_humana.csv').read_bytes()).hexdigest()}
(out/'comparacao.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
print('EXEMPLOS_DIVERGENCIAS')
for r in [r for r in rows if r['situacao']=='divergencia_pendente'][:12]: print(json.dumps(r,ensure_ascii=False))
