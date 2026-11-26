import csv, json, re, sys
from collections import Counter
from datetime import datetime
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path

directory = Path('../GitHub/toledo-portfolio-optimizationn/intel/data_acquisition')
raw_path = directory / 'noticias_b3.json'
csv_path = directory / 'noticias_b3.csv'
original = raw_path.read_bytes()
base = json.loads(original)
csv_original = csv_path.read_bytes()

def normalize(title, source):
    title = title.removesuffix(' - ' + source)
    return ' '.join(re.sub(r'[^\w\s]', ' ', title.casefold()).split())

def identities(ticker, day, title, source, url):
    return {(ticker, day, 'title', normalize(title, source)), (ticker, day, 'url', url.split('?')[0])}

seen = set()
for row in base:
    try:
        day = parsedate_to_datetime(row['data']).date().isoformat()
    except (ValueError, TypeError):
        day = row['janela_inicio']
    seen.update(identities(row['ticker'].upper().removesuffix('.SA'), day, row['titulo'], row['fonte'], row['link_google']))

files = ['coleta_arquivos_veiculos.csv', 'coleta_reuters_direta.csv', 'coleta_multilingue.csv', 'coleta_adr.csv']
candidates = []
audit_path = directory.parents[1] / 'analise_resultados/simulacao_pool/auditoria_pool.csv'
with audit_path.open() as stream:
    pairs = {(x['news_date'], x['asset'].removesuffix('.SA')) for x in csv.DictReader(stream)}
assert len(pairs) == 240
for file in files:
    path = Path('tmp')/file
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            ticker, day = row['ativo'], row['data']
            assert (day, ticker) in pairs and row['publicado_em_utc'][:10] == day
            ids = identities(ticker, day, row['titulo'], row['fonte'], row['url'])
            if ids & seen:
                continue
            seen.update(ids)
            candidates.append(row)
candidates.sort(key=lambda a:(a['data'],a['ativo'],a['titulo'].casefold()))
for n, row in enumerate(candidates,1):
    row['titulo'] = ' '.join(row['titulo'].split())
    row['id'] = n
path=Path('tmp/candidatas_multilingues.csv')
with path.open('w',encoding='utf-8-sig',newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['id',*list(candidates[0])[:-1]])
    writer.writeheader()
    writer.writerows(candidates)
print(len(candidates),len({(x['data'],x['ativo']) for x in candidates}))



