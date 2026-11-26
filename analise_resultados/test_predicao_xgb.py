"""Testes de disponibilidade temporal e definicao das referencias."""
import unittest
import numpy as np
import pandas as pd
from .avaliar_predicao_xgb import baselines, metrics


class PredictionTests(unittest.TestCase):
    def test_future_outcomes_do_not_change_baselines(self):
        actual = pd.Series(np.linspace(-.02,.03,200), index=pd.bdate_range('2024-01-01',periods=200))
        changed = actual.copy()
        # Na posicao 120, com h=21, o ultimo alvo conhecido tem origem em 99.
        changed.iloc[100:] = -100.
        a,b = baselines(actual,21)
        c,d = baselines(changed,21)
        self.assertAlmostEqual(a.iloc[120],c.iloc[120])
        self.assertEqual(b.iloc[120],d.iloc[120])
        changed.iloc[99] = -100.
        c,_ = baselines(changed,21)
        self.assertNotEqual(a.iloc[120],c.iloc[120])

    def test_minimum_observations_and_majority(self):
        actual = pd.Series([.01]*100+[-.01]*100)
        mean, majority = baselines(actual,5)
        self.assertTrue(np.isnan(mean.iloc[83]))
        self.assertTrue(np.isnan(majority.iloc[83]))
        self.assertAlmostEqual(mean.iloc[84],.01)
        self.assertEqual(majority.iloc[84],1)

    def test_skill_and_direction_can_disagree(self):
        frame = pd.DataFrame(dict(actual=[.01,-.01,.02,-.02], pred=[.1,-.1,.2,-.2],
                                  historical=[0.]*4,majority=[1.]*4))
        m=metrics(frame)
        self.assertEqual(m['hr'],100.)
        self.assertLess(m['skill_zero'],0.)
        self.assertAlmostEqual(m['hr_majority'],50.)


if __name__=='__main__':
    unittest.main()
