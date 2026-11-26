import json
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
hosts=['einvestidor.estadao.com.br','www.estadao.com.br','www.exame.com','exame.com','www.bloomberglinea.com.br','www.bloomberglinea.com','www.suno.com.br','www.istoedinheiro.com.br','istoedinheiro.com.br','forbes.com.br','www.poder360.com.br','www.cnnbrasil.com.br','timesbrasil.com.br','www.cnbc.com','www.riotimesonline.com','brazilian.report','www.brasilmineral.com.br','valorinveste.globo.com','www.gazetadopovo.com.br','www.metropoles.com','www.correiodopovo.com.br','www.correio24horas.com.br','www.nsctotal.com.br','www.diariodocomercio.com.br','diariodocomercio.com.br','www.otempo.com.br','www.infomoney.com.br','www.financenews.com.br','financenews.com.br','www.moneyreport.com.br','www.jornalcontabil.com.br','www.mining.com','www.offshore-energy.biz','www.latamlist.com','www.americaeconomia.com','www.df.cl','www.ambito.com','www.eleconomista.es','www.expansion.com','financialpost.com','www.investors.com','www.miningweekly.com','www.worldoil.com','br.investing.com']
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

