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

from itertools import product
import html
sources=dict([
    ('www.infomoney.com.br','InfoMoney'),('www.moneytimes.com.br','Money Times'),
    ('www.seudinheiro.com','Seu Dinheiro'),('neofeed.com.br','NeoFeed'),
    ('www.bloomberglinea.com.br','Bloomberg Línea Brasil'),('www.suno.com.br','Suno'),
    ('www.istoedinheiro.com.br','ISTOÉ DINHEIRO'),('forbes.com.br','Forbes Brasil'),
    ('www.poder360.com.br','Poder360'),('www.cnnbrasil.com.br','CNN Brasil'),
    ('timesbrasil.com.br','Times Brasil | CNBC'),('www.spglobal.com','S&P Global'),
    ('www.finanzen.net','finanzen.net'),('www.offshore-energy.biz','Offshore Energy'),
    ('www.mining.com','Mining.com'),('www.rigzone.com','Rigzone')])
def archive(job):
    host,d=job
    day=date.fromisoformat(d)
    results=[]
    page=1
    while True:
        query=dict(after=f'{day-timedelta(days=1)}T00:00:00',before=f'{day+timedelta(days=1)}T23:59:59',
                   per_page=100,page=page,_fields='title,link,date_gmt')
        url='https://'+host+'/wp-json/wp/v2/posts?'+urlencode(query)
        try:
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=10) as feed:
                posts=json.load(feed)
                pages=int(feed.headers.get('X-WP-TotalPages','1'))
            if not isinstance(posts,list):
                raise ValueError('Arquivo indisponível')
        except (OSError,ValueError) as exc:
            return host,d,str(exc),results
        for post in posts:
            dt=post.get('date_gmt','')
            if not dt or dt[:10]!=d:
                continue
            title=' '.join(html.unescape(re.sub('<[^>]+>',' ',post['title']['rendered'])).split())
            for target_day,t in pairs:
                if target_day!=d or not title or not match(title,t):
                    continue
                if t=='B3SA3' and not re.search(r'\bB3SA3\b|\bB3 (anuncia|lança|registra|aprova|paga|compra|fecha acordo|reporta|faz parceria)|Brasil Bolsa Balcão',title,re.I):
                    continue
                results.append(dict(data=d,ativo=t,status='noticia_encontrada',
                                    status_busca_web='concluida',titulo=title,fonte=sources[host],
                                    publicado_em_utc=dt+'+00:00',url=post['link'],url_original=post['link'],
                                    origem='arquivo_veiculo',consulta_rss=url))
        if page>=pages:
            return host,d,'',results
        page+=1

days=sorted({d for d,t in pairs})
jobs=list(product(sources,days))
fields='data ativo status status_busca_web titulo fonte publicado_em_utc url url_original origem consulta_rss erro_busca'.split()
with Path('tmp/coleta_arquivos_veiculos.csv').open('w',newline='',encoding='utf-8-sig') as f, ThreadPoolExecutor(max_workers=12) as pool:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    stats=Counter()
    for host,day,err,rows in pool.map(archive,jobs):
        stats[host,err.split(':')[0] if err else 'ok']+=1
        w.writerows(rows)
        f.flush()
    print(stats)

