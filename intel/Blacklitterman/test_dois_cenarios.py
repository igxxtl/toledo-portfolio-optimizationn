"""Verificacoes do motor experimental, sem rede ou downloads."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from .simular_dois_cenarios import allocate, project, simulate, subsets, build_signals


class SimulationTests(unittest.TestCase):
    def test_projection_matches_bisection(self):
        rng = np.random.default_rng(8)
        for n in range(3, 21):
            v = rng.normal(size=n)
            lo, hi = v.min() - .4, v.max() - .001
            for _ in range(80):
                mid = (lo + hi) / 2
                if np.clip(v - mid, .001, .4).sum() > 1:
                    lo = mid
                else:
                    hi = mid
            expected = np.clip(v - (lo + hi) / 2, .001, .4)
            np.testing.assert_allclose(project(v, .4), expected, atol=1e-12)

    def test_optimizer_analytic_two_assets(self):
        cov = np.diag([.0004, .0009])
        mu = np.array([.001, .0012])
        risk = 3.
        expected = np.clip((mu[0] - mu[1] + risk * cov[1, 1])
                           / (risk * (cov[0, 0] + cov[1, 1])), .2, .8)
        np.testing.assert_allclose(allocate(mu, cov, .8, risk), [expected, 1 - expected], atol=1e-6)

    def test_close_execution_and_weight_drift(self):
        dates = pd.date_range('2025-01-02', periods=4, freq='B')
        ret = pd.DataFrame([[0, 0], [.5, 0], [.1, 0], [0, .2]], index=dates, columns=['A', 'B'])
        signal = dict(signal_date=dates[0], cov=np.eye(2) * .01, BL=np.zeros(2), Markowitz=np.zeros(2))
        signals = {dates[1]: signal}
        m, curve, _ = simulate(dates, ret, signals, 2, 'BL', (0, 1), 3., 0.)
        rf = 1.15 ** (1 / 252) - 1
        self.assertAlmostEqual(curve.ret.iloc[1], rf)  # retorno de 50% antes da compra nao e capturado
        self.assertAlmostEqual(curve.ret.iloc[2], .05)
        self.assertAlmostEqual(curve.ret.iloc[3], .2 * .5 / 1.05)
        expected = (1 + rf)**2 * (.5 * 1.1 + .5 * 1.2)
        self.assertAlmostEqual(m['return_pct'], (expected - 1) * 100)
        _, charged, trades = simulate(dates, ret, signals, 2, 'BL', (0, 1), 3., 10.)
        self.assertGreater(trades.cost_fraction.iloc[0], 0)
        self.assertLess(charged.capital.iloc[-1], curve.capital.iloc[-1])

    def test_sampling_unique_reproducible_and_exhaustive_edges(self):
        self.assertEqual(len(subsets(1, 20, 30, 2025)), 20)
        self.assertEqual(len(subsets(19, 20, 30, 2025)), 20)
        self.assertEqual(len(subsets(20, 20, 30, 2025)), 1)
        a = subsets(10, 20, 30, 2025)
        self.assertEqual(a, subsets(10, 20, 30, 2025))
        self.assertEqual(len(set(a)), 30)

    def test_future_outcomes_do_not_change_first_decision(self):
        dates = pd.bdate_range('2023-01-02', '2025-03-31')
        rng = np.random.default_rng(12)
        returns = pd.DataFrame(rng.normal(.0002, .01, (len(dates), 2)), index=dates, columns=['A', 'B'])
        prices = 100 * (1 + returns).cumprod()
        predictions = {}
        altered = {}
        first_signal = pd.Timestamp('2025-01-01')
        for a in prices:
            p = dict(h=5, pred=pd.Series(.002, index=dates),
                     actual=np.log(prices[a].shift(-5) / prices[a]),
                     maturity=pd.Series(dates, index=dates).shift(-5))
            predictions[a] = p
            altered[a] = {k: v.copy() if isinstance(v, pd.Series) else v for k, v in p.items()}
            unavailable = p['maturity'].gt(first_signal) | p['maturity'].isna()
            altered[a]['actual'].loc[unavailable] = 99.
            altered[a]['pred'].loc[dates > first_signal] = -99.
        empty_news = pd.DataFrame(columns=['view_date', 'news_count', 'sentiment_signal'])
        with patch('intel.Blacklitterman.simular_dois_cenarios.load_daily_sentiment_for_ticker',
                   return_value=empty_news):
            _, original, _ = build_signals(returns, prices, predictions, first_signal,
                                          pd.Timestamp('2025-01-31'), 126, 3.)
            changed_returns = returns.copy()
            changed_returns.loc[dates > first_signal] = .9
            _, changed, _ = build_signals(changed_returns, prices, altered, first_signal,
                                         pd.Timestamp('2025-01-31'), 126, 3.)
        execution = next(iter(original))
        np.testing.assert_array_equal(original[execution]['BL'], changed[execution]['BL'])
        np.testing.assert_array_equal(original[execution]['Markowitz'], changed[execution]['Markowitz'])


if __name__ == '__main__':
    unittest.main()
