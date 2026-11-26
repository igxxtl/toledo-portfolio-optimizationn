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
with Path('experimentos/noticias_pares_ausentes_2026_11_25/noticias_pares_ausentes.csv').open(encoding='utf-8-sig',newline='') as source:
    pairs=[(x['data'],x['ativo']) for x in csv.DictReader(source) if x['status']=='sem_resultado']
assert len(pairs)==142
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
def search(pair):
    d,t=pair
    day=date.fromisoformat(d)
    terms=[t]+names[t].split('|')
    if t=='B3SA3':
        terms=[t,'Brasil Bolsa Balcão','ações da B3']
    if t=='RAIL3':
        terms=[t,'Rumo Logística','Rumo Malha']
    company=dict(B3SA3='B3 bolsa',VALE3='Vale mineradora',EQTL3='Equatorial Energia',SUZB3='Suzano celulose',LREN3='Lojas Renner',RAIL3='Rumo ferroviária',SANB11='Santander Brasil',PRIO3='PRIO petróleo',RADL3='RD Saúde').get(t,names[t].split('|')[0])
    query=f'("{t}" OR "{company}") (site:infomoney.com.br OR site:moneytimes.com.br OR site:valor.globo.com OR site:exame.com OR site:investing.com OR site:suno.com.br OR site:seudinheiro.com OR site:estadao.com.br OR site:globo.com OR site:cnnbrasil.com.br) after:{day-timedelta(days=1)} before:{day+timedelta(days=2)}'
    url='https://news.google.com/rss/search?'+urlencode(dict(q=query,hl='pt-BR',gl='BR',ceid='BR:pt-419'))
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

out=Path('tmp/segunda_coleta_noticias.csv')
fields='data ativo status status_busca_web titulo fonte publicado_em_utc url url_original origem consulta_rss erro_busca'.split()
with out.open('w',newline='',encoding='utf-8-sig') as f, ThreadPoolExecutor(max_workers=7) as pool:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    for i,((d,t),q,err,web) in enumerate(pool.map(search,pairs),1):
        for row in web:
            w.writerow(dict(data=d,ativo=t,status='noticia_encontrada',status_busca_web='concluida',consulta_rss=q,**row))
        f.flush()
        if i%25==0 or err:
            print(i,err,flush=True)

