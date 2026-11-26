import csv, io, json, os, re, tempfile
from collections import Counter
from datetime import datetime
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path

directory = Path('../GitHub/toledo-portfolio-optimizationn/intel/data_acquisition')
raw_path = directory/'noticias_b3.json'
csv_path = directory/'noticias_b3.csv'
old_raw = raw_path.read_bytes()
old_csv = csv_path.read_bytes()
base = json.loads(old_raw)
reader = csv.DictReader(io.StringIO(old_csv.decode('utf-8-sig'), newline=''))
fields = reader.fieldnames
base_csv = list(reader)
assert [{k:r[k] for k in base[0]} for r in base_csv] == base
def normalize(title, source):
    return ' '.join(re.sub(r'[^\w\s]', ' ', title.removesuffix(' - ' + source).casefold()).split())

def keys(ticker, day, title, source, url):
    return {(ticker, day, 'title', normalize(title,source)), (ticker, day,'url',url.split('?')[0])}
known=set()
for row in base:
    try:
        day=parsedate_to_datetime(row['data']).date().isoformat()
    except (ValueError,TypeError):
        day=row['janela_inicio']
    known.update(keys(row['ticker'].upper().removesuffix('.SA'),day,row['titulo'],row['fonte'],row['link_google']))
skip={1,2,3,4,8,10,25,27,28,32,33,38,39,48,59,69,71,72,74,76,81,86,93,96,100,103,122,123,124}
with Path('tmp/candidatas_ampliacao.csv').open(encoding='utf-8-sig', newline='') as f:
    found=[r for r in csv.DictReader(f) if int(r['id']) not in skip]
assert len(found)+len(skip)==126
bndes='https://agenciadenoticias.bndes.gov.br/noticia/Com-R%24-6078-mi-BNDES-apoia-plano-de-inovacao-da-WEG'
found.append(dict(data='2025-03-31',ativo='WEGE3',titulo='Com R$ 60,78 mi, BNDES apoia plano de inovação da WEG',
                  fonte='Agência BNDES de Notícias',publicado_em_utc='2025-03-31T19:08:00+00:00',url=bndes,url_original=bndes))
audit_path=directory.parents[1]/'analise_resultados/simulacao_pool/auditoria_pool.csv'
with audit_path.open() as f:
    expected={(x['news_date'],x['asset'].removesuffix('.SA')) for x in csv.DictReader(f)}
initial=Path('experimentos/noticias_pares_ausentes_2026_11_25/noticias_pares_ausentes.csv')
with initial.open(encoding='utf-8-sig',newline='') as f:
    missing={(x['data'],x['ativo']) for x in csv.DictReader(f) if x['status']=='sem_resultado'}

additional=[]
with_links=[]
by_pair=set()
sources=Counter()
for r in found:
    day,ticker=r['data'],r['ativo']
    ts=datetime.fromisoformat(r['publicado_em_utc'])
    assert (day,ticker) in expected and ts.date().isoformat()==day
    assert r['url'].startswith('https://')
    ids=keys(ticker,day,r['titulo'],r['fonte'],r['url'])
    if ids & known:
        continue
    known.update(ids)
    by_pair.add((day,ticker))
    sources[r['fonte']]+=1
    title=r['titulo'].removesuffix(' - '+r['fonte'])+' - '+r['fonte']
    item=dict(ticker=ticker,janela_inicio=day,janela_fim=day,titulo=title,fonte=r['fonte'],
              data=format_datetime(ts,usegmt=True),link_google=r['url'])
    additional.append(item)
    with_links.append(dict(item,link_real=r['url_original']))

assert additional and len(expected)==240
raw_suffix = old_raw[len(old_raw.rstrip()):]
raw_updated = (old_raw.rstrip()[:-1].rstrip() + b',\n'
               + json.dumps(additional,ensure_ascii=False,indent=2).encode()[2:-2] + b'\n]' + raw_suffix)
assert json.loads(raw_updated) == base+additional
buffer=io.StringIO(newline='')
csv.DictWriter(buffer,fieldnames=fields).writerows(with_links)
assert old_csv.endswith(b'\r\n')
csv_updated = old_csv + buffer.getvalue().encode()
result=list(csv.DictReader(io.StringIO(csv_updated.decode('utf-8-sig'),newline='')))
assert result == base_csv+with_links
assert [{k:r[k] for k in base[0]} for r in result]==json.loads(raw_updated)

def stage(path,data):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as file:
        file.write(data)
        return Path(file.name)

assert raw_path.read_bytes()==old_raw and csv_path.read_bytes()==old_csv
staged_raw=stage(raw_path,raw_updated)
try:
    staged_csv=stage(csv_path,csv_updated)
    try:
        os.replace(staged_raw,raw_path)
        os.replace(staged_csv,csv_path)
    finally:
        staged_csv.unlink(missing_ok=True)
finally:
    staged_raw.unlink(missing_ok=True)
assert raw_path.read_bytes()==raw_updated and csv_path.read_bytes()==csv_updated
print('novos',len(additional),'pares',len(by_pair),'anteriormente_sem_resultado',len(by_pair&missing),'ainda_sem_resultado',len(missing-by_pair),'total',len(result),'fontes',len(sources))
print(sources.most_common(12))

