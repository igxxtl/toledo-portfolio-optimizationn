"""Causalidade da regressao conjunta e da normalizacao de alvo."""
import unittest
import numpy as np
import pandas as pd
from .experimento_melhoria import masks
from .experimento_metodologia import train


class PooledTests(unittest.TestCase):
    def test_future_labels_do_not_change_pooled_or_normalized_predictions(self):
        rng=np.random.default_rng(2);dates=pd.bdate_range('2022-01-03',periods=800)
        f=pd.DataFrame([dict(date=d,maturity=dates[i+21] if i+21<len(dates) else pd.NaT,
            current=0.,actual=rng.normal(0,.03),vol_21=.01,horizon=21,x=rng.normal(),
            idA=float(a=='A'),idB=float(a=='B')) for i,d in enumerate(dates) for a in ['A','B']])
        tr,te=masks(f,'2024-01-01','2024-07-01','validation')
        changed=f.copy();changed.loc[te,'actual']=1000.
        for normal in [False,True]:
            a,audit=train(f,['x','idA','idB'],'completo',tr,te,normal)
            b,_=train(changed,['x','idA','idB'],'completo',tr,te,normal)
            np.testing.assert_array_equal(a,b)
            self.assertLess(pd.Timestamp(audit['max_train_maturity']),f.date.iloc[te].min())
            self.assertLess(pd.Timestamp(audit['max_fit_maturity']),pd.Timestamp(audit['early_stop_start']))


if __name__=='__main__':unittest.main()
