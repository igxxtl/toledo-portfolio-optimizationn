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
pairs=sorted({(a['news_date'],a['asset'].removesuffix('.SA')) for a in audit })
assert len(pairs)==240
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
aliases=dict(ABEV3='Ambev',B3SA3='B3 stock exchange',BBAS3='Banco do Brasil',BBDC4='Banco Bradesco',CMIG4='Cemig',EQTL3='Equatorial Energia',GGBR4='Gerdau',ITUB4='Itau Unibanco',LREN3='Lojas Renner',MGLU3='Magazine Luiza',PETR3='Petrobras',PETR4='Petrobras',PRIO3='PetroRio',RADL3='Raia Drogasil',RAIL3='Rumo Logistica',SANB11='Santander Brasil',SBSP3='Sabesp',SUZB3='Suzano',VALE3='Vale mining',WEGE3='WEG Brazil')
def search(job):
    pair,edition=job
    d,t=pair
    day=date.fromisoformat(d)
    terms=[t]+names[t].split('|')
    if t=='B3SA3':
        terms=[t,'Brasil Bolsa Balcão','ações da B3']
    if t=='RAIL3':
        terms=[t,'Rumo Logística','Rumo Malha']
    eng = edition=='US'
    spanish = edition=='ES'
    company=aliases[t]
    if spanish:
        company={'B3SA3':'Bolsa de Brasil','VALE3':'Vale minera','WEGE3':'WEG Brasil','RAIL3':'Rumo Brasil'}.get(t,company)
    if not eng and not spanish:
        company={'B3SA3':'B3 bolsa','VALE3':'Vale mineradora','WEGE3':'WEG','RAIL3':'Rumo ferrovia','PRIO3':'PRIO petróleo'}.get(t,names[t].split('|')[0])
    domain = '' if eng or spanish else ' (site:uol.com.br OR site:oglobo.globo.com OR site:valorinveste.globo.com OR site:folha.uol.com.br OR site:poder360.com.br OR site:istoedinheiro.com.br OR site:forbes.com.br OR site:timesbrasil.com.br OR site:neofeed.com.br OR site:agenciabrasil.ebc.com.br OR site:investnews.com.br)'
    query=f'"{adr[t]}" (stock OR shares OR earnings OR dividend) after:{day-timedelta(days=1)} before:{day+timedelta(days=2)}'
    url='https://news.google.com/rss/search?'+urlencode(dict(q=query,hl={'US':'en-US','ES':'es-419','BR':'pt-BR'}[edition],gl={'US':'US','ES':'MX','BR':'BR'}[edition],ceid={'US':'US:en','ES':'MX:es-419','BR':'BR:pt-419'}[edition]))
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
out=Path('tmp/coleta_adr.csv')
fields='data ativo status status_busca_web titulo fonte publicado_em_utc url url_original origem consulta_rss erro_busca'.split()
adr=dict(ABEV3='ABEV',BBDC4='BBD',CMIG4='CIG',GGBR4='GGB',ITUB4='ITUB',PETR3='PBR.A',PETR4='PBR',SBSP3='SBS',SUZB3='SUZ',VALE3='VALE')
jobs=[(pair,'US') for pair in pairs if pair[1] in adr]
with out.open('w',newline='',encoding='utf-8-sig') as f, ThreadPoolExecutor(max_workers=12) as pool:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    for i,((d,t),q,err,web) in enumerate(pool.map(search,jobs),1):
        for row in web:
            w.writerow(dict(data=d,ativo=t,status='noticia_encontrada',status_busca_web='concluida',consulta_rss=q,**row))
        f.flush()
        if i%120==0 or err:
            print(i,err,flush=True)


