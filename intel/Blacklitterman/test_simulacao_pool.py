"""Controles da atualizacao de views e confianca na simulacao conjunta."""
import unittest
import numpy as np
from .simular_pool_vs_atual import fixed_omega_posterior


class PoolSimulationTests(unittest.TestCase):
    def test_view_equal_to_prior_does_not_update(self):
        pi=np.array([.001,.002]);cov=np.array([[.01,.002],[.002,.02]])
        result=fixed_omega_posterior(pi,cov,np.array([.001,.003]),pi.copy(),.02)
        np.testing.assert_array_equal(result,pi)

    def test_uncertain_view_stays_closer_to_prior(self):
        pi=np.array([.001]);cov=np.array([[.01]]);q=np.array([.02])
        confident=fixed_omega_posterior(pi,cov,np.array([.00001]),q,.02)
        uncertain=fixed_omega_posterior(pi,cov,np.array([100.]),q,.02)
        self.assertLess(abs(uncertain[0]-pi[0]),abs(confident[0]-pi[0]))

    def test_cross_covariance_can_transmit_a_single_view(self):
        pi=np.zeros(2);cov=np.array([[.01,.005],[.005,.02]])
        result=fixed_omega_posterior(pi,cov,np.array([.00001,1e6]),np.array([.02,0.]),.02)
        self.assertGreater(result[1],0.)


if __name__=='__main__':unittest.main()
