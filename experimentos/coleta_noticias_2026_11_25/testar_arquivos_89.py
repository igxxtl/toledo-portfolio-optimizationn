import json
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
hosts=['www.gazetasp.com.br','www.gazetadopovo.com.br','www.em.com.br','www.hojeemdia.com.br','www.itatiaia.com.br','www.otempo.com.br','www.tribunapr.com.br','www.folhadelondrina.com.br','www.atribuna.com.br','www.tribunademinas.com.br','tribunademinas.com.br','www.jornalcontabil.com.br','www.abcdoabc.com.br','abcdoabc.com.br','www.jornaldocomercio.com','www.jornaldebrasilia.com.br','jornaldebrasilia.com.br','www.investidor10.com.br','www.tc.com.br','www.bpmoney.com.br','bpmoney.com.br','www.brazilstockguide.com','brazilstockguide.com','brazilenergyinsight.com','www.energiahoje.com.br','eixos.com.br','www.eixos.com.br','www.brasilenergia.com.br','www.canalenergia.com.br','www.petronoticias.com.br','petronoticias.com.br','www.papnews.com','www.pulpapernews.com','www.tissueonlinenews.com','tissueonline.com.br','www.revistafatorbrasil.com.br','www.mining-technology.com','www.railwaygazette.com','www.railjournal.com','www.globalrailwayreview.com','www.railwayage.com','www.offshore-mag.com','splash247.com','www.offshore-technology.com','www.offshore-energy.biz','www.mining.com','www.nasdaq.com','www.etf.com','www.thestreet.com','www.theglobeandmail.com','www.barchart.com','www.fool.com','www.bnamericas.com','www.portafolio.co','www.larepublica.co','www.fundspeople.com','fundspeople.com','www.agendaeconomica.com.br','www.mercadoeconsumo.com.br','mercadoeconsumo.com.br','www.mundodomarketing.com.br','www.abras.com.br','gkpb.com.br','www.revistamercado.com.br','www.revistafullpower.com.br','www.estadaomatogrosso.com.br','www.enfoquems.com.br','www.revistaferroviaria.com.br','revistaferroviaria.com.br','canalrural.com.br','agro.estadao.com.br','www.jota.info']
def probe(host):
    try:
        u='https://'+host+'/wp-json/wp/v2/posts?after=2025-03-31T00%3A00%3A00&before=2025-03-31T23%3A59%3A59&per_page=1&_fields=title,link,date_gmt'
        with urlopen(Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=5) as f:
            x=json.load(f)
            if isinstance(x,list):
                return host, f.headers.get('X-WP-Total'), x[0]['date_gmt'] if x else ''
    except (OSError,ValueError,KeyError):
        pass
with ThreadPoolExecutor(max_workers=15) as pool:
    for result in pool.map(probe,hosts):
        if result:
            print(result,flush=True)


