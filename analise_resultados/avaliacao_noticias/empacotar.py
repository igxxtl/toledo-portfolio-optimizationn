"""Gera HTML unico para abrir localmente, sem servidor nem instalacao."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
html = (HERE / 'index.html').read_text(encoding='utf-8')
for filename in ('noticias.js', 'app.js'):
    source = (HERE / filename).read_text(encoding='utf-8').replace('</script', '<\\/script')
    html = html.replace(f'<script src="{filename}"></script>', '<script>\n' + source + '\n</script>')
(HERE / 'AVALIAR_NOTICIAS.html').write_text(html, encoding='utf-8')
print('Gerado AVALIAR_NOTICIAS.html (300 noticias; sem rotulos do modelo).')
