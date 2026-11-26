import csv, io, json, os, re, tempfile
from datetime import datetime
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path

directory = Path('../GitHub/toledo-portfolio-optimizationn/intel/data_acquisition')
raw_path = directory / 'noticias_b3.json'
csv_path = directory / 'noticias_b3.csv'
source_path = Path('experimentos/noticias_pares_ausentes_2026_11_25/noticias_pares_ausentes.csv')
original = raw_path.read_bytes()
base = json.loads(original)
fields = ['ticker', 'janela_inicio', 'janela_fim', 'titulo', 'fonte', 'data', 'link_google']

def normalized(title, source):
    return ' '.join(re.sub(r'[^\w\s]', ' ', title.removesuffix(' - ' + source).casefold()).split())

def key(row):
    try:
        day = parsedate_to_datetime(row['data']).date().isoformat()
    except (ValueError, TypeError):
        day = row['janela_inicio']
    return row['ticker'].upper().removesuffix('.SA'), day

def identities(row):
    ticker, day = key(row)
    yield ticker, day, 'titulo', normalized(row['titulo'], row['fonte'])
    if row['link_google']:
        yield ticker, day, 'url', row['link_google'].split('?')[0]

seen = {identity for row in base for identity in identities(row)}
added = []
already_present = []
direct_by_identity = {}
with source_path.open(encoding='utf-8-sig', newline='') as stream:
    selected = [row for row in csv.DictReader(stream) if row['status'] == 'noticia_encontrada']
for row in selected:
    published = datetime.fromisoformat(row['publicado_em_utc'])
    assert published.date().isoformat() == row['data']
    news = dict(ticker=row['ativo'], janela_inicio=row['data'], janela_fim=row['data'],
                titulo=f"{row['titulo']} - {row['fonte']}" if row['fonte'] else row['titulo'],
                fonte=row['fonte'], data=format_datetime(published, usegmt=True),
                link_google=row['url'])
    ids = set(identities(news))
    if row['url_original']:
        direct_by_identity.update(dict.fromkeys(ids, row['url_original']))
    if ids & seen:
        already_present.append(row)
        continue
    added.append(news)
    seen.update(ids)

assert len(selected) == len(added) + len(already_present) == 105
assert len({(row['data'], row['ativo']) for row in selected}) == 58
consolidated = base + added
assert all(set(row) == set(fields) for row in consolidated)
stripped = original.rstrip()
assert stripped.endswith(b']')
suffix = original[len(stripped):]
appendix = json.dumps(added, ensure_ascii=False, indent=2).encode('utf-8')[2:-2]
updated = stripped[:-1].rstrip() + b',\n' + appendix + b'\n]' + suffix if added else original
assert json.loads(updated) == consolidated
buffer = io.StringIO(newline='')
writer = csv.DictWriter(buffer, fieldnames=fields + ['link_real'])
writer.writeheader()
writer.writerows(dict(row, link_real=next((direct_by_identity[i] for i in identities(row) if i in direct_by_identity), ''))
                 for row in consolidated)
csv_bytes = buffer.getvalue().encode('utf-8-sig')
parsed = list(csv.DictReader(io.StringIO(csv_bytes.decode('utf-8-sig'), newline='')))
assert [{field: row[field] for field in fields} for row in parsed] == consolidated

def atomic_write(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        stream.write(value)
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)

assert raw_path.read_bytes() == original
assert not csv_path.exists() or csv_path.read_bytes() == csv_bytes
atomic_write(raw_path, updated)
atomic_write(csv_path, csv_bytes)
assert raw_path.read_bytes() == updated and csv_path.read_bytes() == csv_bytes
print(f'Original: {len(base)} | Adicionados: {len(added)} | Já presentes: {len(already_present)} | Consolidado: {len(consolidated)}')

