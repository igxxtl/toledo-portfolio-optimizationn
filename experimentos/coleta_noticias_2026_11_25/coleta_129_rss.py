import csv,json,re,time
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date,timedelta,timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode
from urllib.request import Request,urlopen
import xml.etree.ElementTree as ET
r=Path('../GitHub/toledo-portfolio-optimizationn')
with (r/'analise_resultados/simulacao_pool/auditoria_pool.csv').open() as f:
    audit=list(csv.DictReader(f))
pairs=sorted({(a['news_date'],a['asset'].removesuffix('.SA')) for a in audit if int(a['news_count'])==0})

data=json.loads((r/'intel/data_acquisition/noticias_b3.json').read_text(encoding='utf-8'))
known={(parsedate_to_datetime(x['data']).date().isoformat(),x['ticker']) for x in data[7835:]}
pairs=[x for x in pairs if x not in known]
assert len(pairs)==129

names=dict(zip(sorted({a['asset'].removesuffix('.SA') for a in audit}),['Ambev','B3','Banco do Brasil','Bradesco','Cemig','Equatorial','Gerdau','Itaú|Itau','Renner','Magazine Luiza|Magalu','Petrobras|PETR4','Petrobras|PETR3','PRIO|PetroRio|Petro Rio','Raia Drogasil|Drogasil|RD Saúde','Rumo','Santander','Sabesp','Suzano','Vale','WEG']))
def match(title,t):
    terms=[t]+names[t].split('|')
    if t=='B3SA3':
        terms=[t,'Brasil Bolsa Balcão']
        if re.search(r'\bB3 (anuncia|lança|registra|aprova|paga|compra|fecha acordo|reporta|faz parceria)',title,re.I):
            return True
    return any(re.search(r'(?<!\w)'+re.escape(x)+r'(?!\w)',title,0 if x in ('Vale','Rumo','WEG','PRIO') else re.I) for x in terms)
def record(title,source,when,url,origin,direct=''):
    stamp=parsedate_to_datetime(when).astimezone(timezone.utc)
    return dict(titulo=title.removesuffix(' - '+source).strip(),fonte=source,publicado_em_utc=stamp.isoformat(),url=url,url_original=direct,origem=origin)
local=defaultdict(list)
for file in ('noticias_b3_sem_duplicados_com_link_real.json','noticias_b3.json'):
    for x in json.loads((r/'intel/data_acquisition'/file).read_text(encoding='utf-8')):
        direct=x.get('link_real') or ''
        try:
            row=record(x['titulo'],x.get('fonte') or '',x.get('data'),x.get('link_google') or '', 'acervo_local',direct if 'news.google.com' not in direct else '')
        except (ValueError,TypeError):
            continue
        for d,t in pairs:
            if row['url'] and row['publicado_em_utc'][:10]==d and match(row['titulo'],t):
                local[d,t].append(row)
def search(job):
    pair,edition=job
    d,t=pair
    day=date.fromisoformat(d)
    terms=[t]+names[t].split('|')
    if t=='B3SA3':
        terms=[t,'Brasil Bolsa Balcão','ações da B3']
    if t=='RAIL3':
        terms=[t,'Rumo Logística','Rumo Malha']
    company={'B3SA3':'Brasil Bolsa Balcão','VALE3':'Vale mineradora','EQTL3':'Equatorial Energia','SUZB3':'Suzano Papel Celulose','LREN3':'Lojas Renner','RAIL3':'Rumo ferrovia','SANB11':'Santander Brasil','PRIO3':'PetroRio','RADL3':'Raia Drogasil','WEGE3':'WEG indústria'}.get(t,names[t].split('|')[0])
    if edition=='BR':
        query=f'{company} (bolsa OR ações OR empresa OR investimentos OR receita OR operação) after:{day} before:{day+timedelta(days=1)}'
    elif edition=='GB':
        alias={'B3SA3':'B3 Brazil exchange','VALE3':'Vale iron ore','WEGE3':'WEG Brazilian','SUZB3':'Suzano pulp','RAIL3':'Rumo Brazil railway','PRIO3':'PRIO Brazil oil','ITUB4':'Itau Unibanco'}.get(t,company)
        query=f'{alias} after:{day} before:{day+timedelta(days=1)}'
    else:
        alias={'WEGE3':'WEG Brasil','VALE3':'Vale minera','B3SA3':'bolsa brasileña B3','RAIL3':'Rumo ferrocarril','SUZB3':'Suzano celulosa','PRIO3':'PetroRio Brasil'}.get(t,company)
        query=f'{alias} after:{day} before:{day+timedelta(days=1)}'
    url='https://news.google.com/rss/search?'+urlencode(dict(q=query,hl={'BR':'pt-BR','GB':'en-GB','ES':'es-ES'}[edition],gl=edition,ceid={'BR':'BR:pt-419','GB':'GB:en','ES':'ES:es'}[edition]))
    for n in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=12) as feed:
                xml=ET.fromstring(feed.read())
            if xml.find('channel') is None:
                raise ValueError('RSS sem canal')
            break
        except (OSError,ValueError,ET.ParseError) as e:
            if n==2:
                return pair,url,str(e),[]
            time.sleep(2**n)
    found=[]
    for e in xml.findall('channel/item'):
        try:
            row=record(e.findtext('title',''),e.findtext('source',''),e.findtext('pubDate'),e.findtext('link',''),'google_news_rss')
        except (ValueError,TypeError):
            continue
        if row['url'] and row['publicado_em_utc'][:10]==d and match(row['titulo'],t):
            found.append(row)
    return pair,url,'',found

from itertools import product
fields='data ativo status status_busca_web titulo fonte publicado_em_utc url url_original origem consulta_rss erro_busca'.split()
with Path('tmp/coleta_129_rss.csv').open('w',newline='',encoding='utf-8-sig') as f, ThreadPoolExecutor(max_workers=10) as pool:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    for i,((d,t),q,err,web) in enumerate(pool.map(search,product(pairs,['BR','GB','ES'])),1):
        for row in web:
            w.writerow(dict(data=d,ativo=t,status='noticia_encontrada',status_busca_web='concluida',consulta_rss=q,**row))
        f.flush()
        if i%129==0 or err:
            print(i,err,flush=True)

