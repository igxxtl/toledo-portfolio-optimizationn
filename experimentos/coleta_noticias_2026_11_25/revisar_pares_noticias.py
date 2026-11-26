import csv
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
p=Path('experimentos/noticias_pares_ausentes_2026_11_25/noticias_pares_ausentes.csv')
with p.open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f)
    fields=reader.fieldnames
    rows=list(reader)
reject=set(range(2,12))|{13,14,16,20,21,22,24,26,27,28,29,32,37,39,40,41,45,46,47,49,50,55,56,58,59,63,64,66,67,68,69,70,87,91,94,97,115,118,122,127,129,134,139,140,147,151,152,153,154,156,157,160,167,168,169,170,172}
group=defaultdict(list)
n=0
for row in rows:
    key=(row['data'],row['ativo'])
    group[key].append(row)
    if row['titulo']:
        n+=1
        row['manter']=n not in reject
        row['titulo']=' '.join(row['titulo'].split())
assert n==172
output=[]
for key, original in group.items():
    kept=[x for x in original if x.get('manter')]
    if kept:
        for x in kept:
            x.pop('manter')
            output.append(x)
    else:
        x={field:original[0][field] for field in ('data','ativo','status_busca_web','consulta_rss','erro_busca')}
        x['status']='sem_resultado' if x['status_busca_web']=='concluida' else 'busca_incompleta'
        output.append(x)
r=Path('../GitHub/toledo-portfolio-optimizationn/analise_resultados/simulacao_pool/auditoria_pool.csv')
with r.open() as f:
    expected={(x['news_date'],x['asset'].removesuffix('.SA')) for x in csv.DictReader(f) if int(x['news_count'])==0}
assert set(group)==expected and len(expected)==200
assert all(datetime.fromisoformat(x['publicado_em_utc']).date().isoformat()==x['data'] and x['url'].startswith('https://') for x in output if x.get('titulo'))
assert len({(x['data'],x['ativo'],x.get('titulo')) for x in output})==len(output)
tmp=p.with_suffix('.tmp')
with tmp.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows(output)
tmp.replace(p)
print('Pares',Counter({s:len({(x['data'],x['ativo']) for x in output if x['status']==s}) for s in {x['status'] for x in output}}))
print('Origens',Counter(x.get('origem') for x in output))
print('Ativos com noticias',len({x['ativo'] for x in output if x.get('titulo')}))
print('Data range',min(x['data'] for x in output),max(x['data'] for x in output))

