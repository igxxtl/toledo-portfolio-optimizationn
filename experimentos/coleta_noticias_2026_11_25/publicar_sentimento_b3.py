import csv, io,json,os,sys,tempfile
from pathlib import Path
from collections import Counter
from datetime import date

sys.dont_write_bytecode=True
repo=Path('../GitHub/toledo-portfolio-optimizationn').resolve()
sys.path.insert(0,str(repo))
from intel.Blacklitterman.sentiment import load_daily_sentiment_for_ticker
from intel.data_acquisition.config import NEWS_WITH_SENTIMENT, NEWS_DEDUPED

candidate=Path('tmp/sentimento_b3/classificadas.json')
rows=json.loads(candidate.read_text(encoding='utf-8'))
old=json.loads(NEWS_WITH_SENTIMENT.read_text(encoding='utf-8'))
assert len(rows)==5044 and len(old)==4681
counts=Counter(row['sentimento'] for row in rows)
assert set(counts)=={'positivo','negativo','neutro'}
old_keys={(r['ticker'],r['titulo'].strip()):r['sentimento'].strip().lower() for r in old}
assert all(old_keys[r['ticker'],r['titulo']]==r['sentimento'] for r in rows[:len(old)])
assert set(old_keys)=={(r['ticker'],r['titulo']) for r in rows[:len(old)]}

with (repo/'analise_resultados/simulacao_pool/auditoria_pool.csv').open() as source:
    audit=list(csv.DictReader(source))
assets=sorted({r['asset'] for r in audit})
assert len(audit)==240 and len(assets)==20
before={a:load_daily_sentiment_for_ticker(a) for a in assets}
after={a:load_daily_sentiment_for_ticker(a,news_path=candidate) for a in assets}
def news_count(frame, day):
    match=frame.loc[frame['view_date'].dt.date.eq(date.fromisoformat(day)),'news_count']
    return int(match.iloc[0]) if len(match) else 0
covered=set()
for row in audit:
    a,d=row['asset'],row['news_date']
    old_count=news_count(before[a],d)
    new_count=news_count(after[a],d)
    assert old_count==int(row['news_count']) and new_count>=old_count
    if not old_count and new_count:
        covered.add((d,a))

schema=['ticker','janela_inicio','janela_fim','titulo','fonte','data','link_google']
def project(row):
    projected={k:row[k] for k in schema}
    if row.get('link_real'):
        projected['link_real']=row['link_real']
    return projected
clean=[project(r) for r in rows]
classified=[dict(r,sentimento=source['sentimento']) for r,source in zip(clean,rows)]
csv_path=NEWS_WITH_SENTIMENT.with_suffix('.csv')
if csv_path.exists():
    raise FileExistsError(csv_path)
csv_buffer=io.StringIO(newline='')
writer=csv.DictWriter(csv_buffer,fieldnames=[*schema,'link_real','sentimento'])
writer.writeheader()
writer.writerows(rows)
csv_content=csv_buffer.getvalue()
assert list(csv.DictReader(io.StringIO(csv_content,newline='')))==rows
targets={
    NEWS_DEDUPED:json.dumps(clean,ensure_ascii=False,indent=2).encode('utf-8'),
    NEWS_WITH_SENTIMENT:json.dumps(classified,ensure_ascii=False,indent=2).encode('utf-8'),
    csv_path:csv_content.encode('utf-8-sig')
}
staged={}
try:
    for target,content in targets.items():
        with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as temp:
            staged[target]=Path(temp.name)
            temp.write(content)
    for target,temp in staged.items():
        os.replace(temp,target)
finally:
    for temp in staged.values():
        temp.unlink(missing_ok=True)
assert all(p.read_bytes()==b for p,b in targets.items())
for asset in assets:
    assert load_daily_sentiment_for_ticker(asset).equals(after[asset])
print(dict(counts),'pares_novos',len(covered),'pares_cobertos',sum(int(x['news_count'])>0 for x in audit)+len(covered))

