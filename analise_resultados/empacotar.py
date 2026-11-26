"""Sincroniza fontes editaveis e monta a entrega do gate registrado no manifesto."""
from pathlib import Path
import argparse
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TCC = Path(r'C:\Users\igor.toledo_visagio\Desktop\TCC\05_Resultados')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync-only', action='store_true')
    args = parser.parse_args()
    work = ROOT/'analise_resultados'
    source = ROOT/'intel/Blacklitterman/outputs/dois_cenarios'
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    gate = int(manifest['bl_parameters']['NEWS_COUNT_MIN'])
    latex = work/'latex'
    chapter = (TCC/'resultados.tex').read_text(encoding='utf-8')
    (latex/'chapter.tex').write_text(chapter.replace('05_Resultados/figuras/', 'figuras/'), encoding='utf-8')
    for p in (TCC/'figuras').glob('*.pdf'):
        shutil.copy2(p,latex/'figuras'/p.name)
    shutil.copy2(work/'LEIA_ME.txt',TCC/'LEIA_ME.txt')
    (TCC/'dados').mkdir(exist_ok=True)
    for p in (work/'dados').iterdir():
        if p.is_file(): shutil.copy2(p,TCC/'dados'/p.name)
    if args.sync_only:
        print('Fontes LaTeX sincronizadas, gate',gate)
        return
    check = json.loads((work/'verificacao_capitulo.json').read_text(encoding='utf-8'))
    assert check['status']=='passed' and check['news_gate']==gate
    shutil.copy2(work/'verificacao_capitulo.json',TCC/'verificacao_capitulo.json')
    pdf = ROOT/f'output/pdf/capitulo_4_resultados_gate{gate}.pdf'
    shutil.copy2(ROOT/'output/pdf/capitulo_4_resultados.pdf',pdf)
    archive = work/f'pacote_capitulo_4_resultados_gate{gate}.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pdf,pdf.name)
        z.write(source/'manifest.json',f'manifesto_simulacao_gate{gate}.json')
        z.write(source/'resumo.csv',f'resumo_simulacao_gate{gate}.csv')
        z.write(ROOT/'intel/Blacklitterman/outputs/sensibilidade_gate_1a5/resumo.csv','analises_apoio/gates_1a5.csv')
        for p in TCC.rglob('*'):
            if p.is_file() and (p.suffix!='.pdf' or p.parent.name=='figuras'):
                z.write(p,'05_Resultados/'+p.relative_to(TCC).as_posix())
        for p in latex.rglob('*'):
            if p.is_file(): z.write(p,'latex_independente/'+p.relative_to(latex).as_posix())
        for name in ['consolidar.py','gerar_capitulo.py','verificar_capitulo.py','empacotar.py',
                     'manifesto_capitulo.json','verificacao_capitulo.json','LEIA_ME.txt']:
            z.write(work/name,'reproducao/'+name)
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    shutil.copy2(archive,work/'pacote_capitulo_4_resultados.zip')
    print('Entrega:',archive)


if __name__=='__main__':
    main()
