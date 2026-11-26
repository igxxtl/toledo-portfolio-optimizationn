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
raw=json.loads((r/'intel/data_acquisition/noticias_b3.json').read_text(encoding='utf-8'))
already={(parsedate_to_datetime(x['data']).date().isoformat(),x['ticker']) for x in raw[7835:]}
pairs=[(a['news_date'],a['asset'].removesuffix('.SA')) for a in audit if int(a['news_count'])==0 and (a['news_date'],a['asset'].removesuffix('.SA')) not in already]
assert len(pairs)==89
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
sources={'www.bpmoney.com.br':'BP Money','jornaldebrasilia.com.br':'Jornal de Brasília',
         'eixos.com.br':'eixos','petronoticias.com.br':'Petronotícias','tissueonline.com.br':'Tissue Online',
         'mercadoeconsumo.com.br':'Mercado & Consumo','revistaferroviaria.com.br':'Revista Ferroviária',
         'canalrural.com.br':'Canal Rural','brazilstockguide.com':'Brazil Stock Guide','www.papnews.com':'Paper First',
         'fundspeople.com':'FundsPeople','elperiodicodelaenergia.com':'El Periódico de la Energía',
         'www.energiaestrategica.com':'Energía Estratégica','energynews.pro':'EnergyNews',
         'splash247.com':'Splash247','gkpb.com.br':'GKPB','www.jornalcontabil.com.br':'Jornal Contábil'}
def archive(job):
    host,d=job
    day=date.fromisoformat(d)
    results=[]
    page=1
    while True:
        query=dict(after=f'{day-timedelta(days=1)}T00:00:00',before=f'{day+timedelta(days=1)}T23:59:59',
                   per_page=100,page=page,_fields='title,link,date_gmt,content')
        url='https://'+host+'/wp-json/wp/v2/posts?'+urlencode(query)
        try:
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=10) as feed:
                posts=json.load(feed)
                pages=int(feed.headers.get('X-WP-TotalPages','1'))
                if pages>20: raise ValueError('Filtro de data ignorado')
            if not isinstance(posts,list):
                raise ValueError('Arquivo indisponível')
        except (OSError,ValueError) as exc:
            return host,d,str(exc),results
        for post in posts:
            dt=post.get('date_gmt','')
            if not dt or dt[:10]!=d:
                continue
            title=' '.join(html.unescape(re.sub('<[^>]+>',' ',post['title']['rendered'])).split())
            body=post.get('content',{}).get('rendered','')
            parts=[html.unescape(re.sub('<[^>]+>',' ',p)) for p in re.findall(r'<p\b[^>]*>(.*?)</p>',body,re.I|re.S)]
            market=r'ações|dividend|balanço|resultad|invest|contrat|receita|lucro|ranking|produção|expans|recomend|dívida|fatur|financia|stock|profit|revenue|earnings|inversi|adquis'
            for target_day,t in pairs:
                if target_day!=d or not title:
                    continue
                title_match=match(title,t)
                specific=names[t].split('|')
                specific={'B3SA3':['Brasil Bolsa Balcão'],'VALE3':['mineradora Vale','Vale Base Metals'],'EQTL3':['Equatorial Energia'],
                          'SUZB3':['Suzano Papel','Suzano celulose'],'RAIL3':['Rumo Logística'],'LREN3':['Lojas Renner'],
                          'SANB11':['Santander Brasil'],'RADL3':['Raia Drogasil','RD Saúde'],'PRIO3':['PetroRio']}.get(t,specific)
                trigger=r'(?<!\w)('+ '|'.join(map(re.escape,[t]+specific)) +r')(?!\w)'
                mentions=[' '.join(p.split()) for p in parts if len(p)>50 and re.search(trigger,p,re.I) and re.search(market,p,re.I)]
                if t=='B3SA3':
                    title_match=bool(re.search(r'\bB3SA3\b|\bB3 (anuncia|lança|registra|aprova|paga|compra|fecha acordo|reporta|faz parceria)|Brasil Bolsa Balcão',title,re.I))
                if not title_match and not mentions:
                    continue
                results.append(dict(data=d,ativo=t,status='noticia_encontrada',
                                    status_busca_web='concluida',titulo=title,fonte=sources[host],
                                    publicado_em_utc=dt+'+00:00',url=post['link'],url_original=post['link'],
                                    origem='arquivo_veiculo',consulta_rss=url,
                                    contexto=('TÍTULO' if title_match else ' | '.join(mentions)[:850])))
        if page>=pages:
            return host,d,'',results
        page+=1

days=sorted({d for d,t in pairs})
jobs=list(product(sources,days))
fields='data ativo status status_busca_web titulo fonte publicado_em_utc url url_original origem consulta_rss erro_busca contexto'.split()
with Path('tmp/coleta_89_arquivos.csv').open('w',newline='',encoding='utf-8-sig') as f, ThreadPoolExecutor(max_workers=12) as pool:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    stats=Counter()
    for host,day,err,rows in pool.map(archive,jobs):
        stats[host,err.split(':')[0] if err else 'ok']+=1
        w.writerows(rows)
        f.flush()
    print(stats)




