"""Testes de causalidade temporal e selecao de features sem olhar o teste."""
import unittest
import numpy as np
import pandas as pd
from .features_experimento import breadth_frames,extra_features,select_columns


class FeatureTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(12);dates=pd.bdate_range('2020-01-01',periods=350)
        close=100*np.exp(np.cumsum(rng.normal(0,.01,len(dates))))
        raw=pd.DataFrame(dict(date=dates,close=close,open=close*.998,high=close*1.01,low=close*.99,
            ibov_close=120*np.exp(np.cumsum(rng.normal(0,.008,len(dates)))),usdbrl_close=np.linspace(4,5,len(dates))))
        prices=pd.DataFrame({str(i):close*(1+i*.01)*np.exp(np.cumsum(rng.normal(0,.005,len(dates)))) for i in range(20)},index=dates)
        return raw,prices

    def test_future_prices_cannot_change_past_features(self):
        raw,prices=self.data();original,_,_=extra_features(raw,breadth_frames(prices))
        changed=raw.copy()
        changed.loc[250:,['close','open','high','low','ibov_close','usdbrl_close']]*=5
        changed_prices=prices.copy();changed_prices.iloc[250:]*=7
        altered,_,_=extra_features(changed,breadth_frames(changed_prices))
        pd.testing.assert_frame_equal(original.iloc[:250],altered.iloc[:250])

    def test_correlation_filter_only_uses_fit_rows(self):
        rng=np.random.default_rng(10);a=rng.normal(size=100);b=rng.normal(size=100)
        X=pd.DataFrame(dict(a=a,b=b,duplicate=2*a,constant=1.))
        before,_=select_columns(X,np.arange(50),list(X))
        X.loc[50:,'b']=X.loc[50:,'a']
        after,_=select_columns(X,np.arange(50),list(X))
        self.assertEqual(before,after);self.assertEqual(before,['a','b'])

    def test_invalid_candle_is_missing_not_infinite(self):
        raw,prices=self.data();raw.loc[200,['open','high','low']]=0.
        features,_,_=extra_features(raw,breadth_frames(prices))
        self.assertTrue(np.isnan(features.loc[200,'ex_gap']))
        self.assertFalse(np.isinf(features.to_numpy()).any())


if __name__=='__main__':unittest.main()
