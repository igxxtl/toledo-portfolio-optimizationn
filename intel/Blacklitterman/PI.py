import numpy as np
import pandas as pd
import yfinance as yf

ativos = ['VALE3.SA', 'BBAS3.SA', 'ITUB3.SA', 'BBDC3.SA', 'ABEV3.SA']
start_date = '2023-01-01'
end_date = '2025-12-31'

#--------------------------------
# MARKET CAP WEIGHTS
#--------------------------------
weights = {}

for t in ativos:
    info = yf.Ticker(t).info
    weights[t] = info.get("marketCap")

market_cap = sum(pd.Series(weights).values)

for i in pd.Series(weights).keys():
    weights[i] = (weights[i] / market_cap)

# print(data)

#--------------------------------
# Retornos (gama)
#--------------------------------

serie_retornos = yf.download(ativos, start=start_date, end=end_date)['Close']

retornos = np.log(serie_retornos / serie_retornos.shift(1)).dropna()

sigma = retornos.cov()

#print(sigma)

#--------------------------------
# Aversão ao risco (lambda)
#--------------------------------

risk_free_rate = 0.15
risk_free_rate_daily = (1 + risk_free_rate)**(1/252) - 1

r_m = retornos @ pd.Series(weights)

delta = (r_m.mean() - risk_free_rate_daily) / r_m.var()

# print(delta)

#--------------------------------
# PI (Retornos implícitos de equilíbrio)
#--------------------------------

pi = delta * sigma @ pd.Series(weights)

print(pi)