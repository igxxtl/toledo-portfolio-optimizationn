"""Verifica separacao por maturidade e invariancia ao alvo futuro."""
import unittest
import numpy as np
import pandas as pd
from .experimento_melhoria import masks,fit_predict,CONFIGS,metric


class TemporalTests(unittest.TestCase):
    def frame(self):
        dates=pd.bdate_range('2022-01-03',periods=800)
        return pd.DataFrame(dict(date=dates,maturity=pd.Series(dates).shift(-21),current=0.))

    def test_training_targets_finish_before_prediction(self):
        frame=self.frame()
        tr,te=masks(frame,'2024-01-01','2024-07-01','validation')
        self.assertTrue((frame.maturity.iloc[tr]<pd.Timestamp('2024-01-01')).all())
        self.assertTrue((frame.maturity.iloc[te]<pd.Timestamp('2025-01-01')).all())
        self.assertGreater(len(te),50)

    def test_future_labels_cannot_change_fitted_predictions(self):
        rng=np.random.default_rng(42);frame=self.frame()
        X=rng.normal(size=(len(frame),5)).astype('float32');y=rng.normal(size=len(frame))*.02
        tr,te=masks(frame,'2024-01-01','2024-07-01','validation')
        a,audit=fit_predict(X,y,frame,tr,te,CONFIGS[0],20,1)
        changed=y.copy();changed[te]=1000.
        b,_=fit_predict(X,changed,frame,tr,te,CONFIGS[0],20,1)
        np.testing.assert_array_equal(a,b)
        self.assertLess(pd.Timestamp(audit['max_fit_maturity']),pd.Timestamp(audit['early_stop_start']))
        self.assertLess(pd.Timestamp(audit['max_train_maturity']),frame.date.iloc[te].min())

    def test_skill_interprets_baseline(self):
        y=np.array([.01,-.02,.03,-.01])
        self.assertEqual(metric(y,np.zeros(4))['skill_zero'],0.)
        self.assertEqual(metric(y,y)['skill_zero'],1.)


if __name__=='__main__':unittest.main()
