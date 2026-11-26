import csv,json,os,sys,re,unicodedata
from pathlib import Path
from collections import Counter
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from dotenv import dotenv_values
from openai import OpenAI

sys.dont_write_bytecode=True
repo=Path('../GitHub/toledo-portfolio-optimizationn').resolve()
sys.path.insert(0,str(repo))
from intel.data_acquisition.config import NEWS_WITH_SENTIMENT,NEWS_DEDUPED,SENTIMENT_MODEL
from intel.data_acquisition.limpa_noticias import title_is_relevant
from intel.data_acquisition.limpa_duplicados import normalize_title
from intel.data_acquisition.analise_sentimento import classify_headline

out=Path('tmp/sentimento_b3')
out.mkdir(exist_ok=True)
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
raw=read(NEWS_WITH_SENTIMENT.with_name('noticias_b3.csv'))
existing=json.loads(NEWS_WITH_SENTIMENT.read_text(encoding='utf-8'))
assert len(raw)==8259 and len(existing)==4681
valid={'positivo','negativo','neutro'}
old={(x['ticker'],x['titulo'].strip()):x['sentimento'].strip().lower() for x in existing if x['sentimento'].strip().lower() in valid}
stats=Counter()
context={}
for file in ('candidatas_129.csv','candidatas_89.csv'):
    for x in read(Path('tmp')/file):
        if x['contexto'] not in ('TÍTULO','RSS',''):
            context[x['ativo'],x['url']]=x['contexto']
            if x['ativo']=='PETR3' and 'EWZ, principal ETF' in x['titulo']:
                context['VALE3',x['url']]=x['contexto']
def day(row):
    try:
        return parsedate_to_datetime(row['data']).date().isoformat()
    except (ValueError,TypeError):
        return datetime.fromisoformat(row['janela_inicio']).date().isoformat()
def company(ticker):
    return ''.join(re.findall('[A-Z]',ticker))[:4]
def keys(row):
    group=(company(row['ticker']),day(row))
    title=normalize_title(row['titulo'].removesuffix(' - '+row['fonte'])).casefold()
    yield group+('title',unicodedata.normalize('NFKC',title))
    for field in ('link_google','link_real'):
        if row.get(field):
            yield group+('url',row[field].partition('?')[0].lower())

prepared=[]
seen_legacy=set()
seen_company=set()
for i,r in enumerate(raw):
    row={k:(v or '').strip() for k,v in r.items()}
    row['ticker']=row['ticker'].upper().removesuffix('.SA')
    title=normalize_title(row['titulo'])
    if not title or not row['ticker']:
        stats['invalida']+=1
        continue
    if i<7835:
        if title in seen_legacy:
            stats['duplicata_legada']+=1
            continue
        seen_legacy.add(title)
        if not title_is_relevant(row['ticker'],row['titulo']):
            stats['titulo_generico_legado']+=1
            continue
    ids=set(keys(row))
    if i>=7835 and ids & seen_company:
        stats['duplicata_complementar']+=1
        continue
    seen_company.update(ids)
    extra=context.get((row['ticker'],row['link_google']),'') if i>=7835 else ''
    prepared.append((i,row,extra))
print('tratamento',dict(stats),'mantidas',len(prepared),'contexto',sum(bool(x[2]) for x in prepared),flush=True)

cachefile=out/'rotulos_novos.jsonl'
cache={}
if cachefile.exists():
    for line in cachefile.read_text(encoding='utf-8').splitlines():
        k,v=json.loads(line)
        if v in valid:
            cache[tuple(k)]=v
pending={}
for i,row,extra in prepared:
    k=(row['ticker'],row['titulo'])
    sentiment_key=(*k,extra)
    if (not extra and k in old) or sentiment_key in cache:
        continue
    pending[sentiment_key]=None
print('cache_legado',len(old),'cache_novo',len(cache),'pendentes',len(pending),'modelo',SENTIMENT_MODEL,flush=True)

def classify(client,k):
    ticker,title,extra=k
    if extra:
        title += '\n\nTrecho da matéria referente ao ativo (use como contexto):\n'+extra
    label=classify_headline(client,ticker,title).strip().lower()
    if label not in valid:
        raise ValueError('Rótulo inválido')
    return label
if pending:
    key=os.getenv('OPENAI_API_KEY') or dotenv_values(repo/'.env').get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('Chave do classificador indisponível')
    client=OpenAI(api_key=key,timeout=40,max_retries=2)
    def saved(f,k,v):
        cache[k]=v
        f.write(json.dumps([k,v],ensure_ascii=False)+'\n')
        f.flush()
    with cachefile.open('a',encoding='utf-8') as f:
        probe=next(iter(pending))
        try:
            saved(f,probe,classify(client,probe))
        except Exception as exc:
            print(type(exc).__name__,getattr(exc,'status_code',None),getattr(exc,'code',None),flush=True)
            sys.exit(2)
        del pending[probe]
        failed=Counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures={pool.submit(classify,client,k):k for k in pending}
            for n,future in enumerate(as_completed(futures),1):
                k=futures[future]
                try:
                    saved(f,k,future.result())
                except Exception as exc:
                    failed[(type(exc).__name__,getattr(exc,'status_code',None),getattr(exc,'code',None))]+=1
                if n%25==0:
                    print('progresso',n,len(pending),dict(failed),flush=True)
        if failed:
            print('falhas',dict(failed),flush=True)
            sys.exit(3)
rows=[]
for i,row,extra in prepared:
    k=(row['ticker'],row['titulo'])
    label=cache.get((*k,extra)) or (old.get(k) if not extra else None)
    assert label in valid
    rows.append(dict(row,sentimento=label))
assert len(raw)==len(rows)+sum(stats.values())
(out/'classificadas.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('sentimento',dict(Counter(x['sentimento'] for x in rows)),'novas',sum(i>=7835 for i,_,_ in prepared),flush=True)

