"""Testes das metricas de cobertura (sem executar o BL completo)."""
import unittest

import pandas as pd

try:
    from intel.Blacklitterman.metricas_cobertura_noticias import calcular_metricas_cobertura
except ImportError:
    from metricas_cobertura_noticias import calcular_metricas_cobertura


class TestMetricasCobertura(unittest.TestCase):
    def setUp(self):
        self.q = pd.DataFrame({
            "view_date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "ticker": ["AAAA3", "BBBB3", "AAAA3", "BBBB3"],
            "news_count": [0, 2, 3, 1],
        })

    def test_n_2_contagens_e_denominadores(self):
        m = calcular_metricas_cobertura(self.q, 2)
        self.assertEqual(m["observacoes_total"], 4)
        self.assertEqual(m["observacoes_elegiveis"], 2)
        self.assertEqual(m["pct_elegiveis"], 50.0)
        self.assertEqual(m["pct_ativo_dias_gt_n"], 25.0)
        self.assertEqual(m["noticias_total_ocorrencias"], 6)
        self.assertEqual(m["noticias_dentro_padrao"], 5)
        self.assertAlmostEqual(m["pct_noticias_dentro_padrao"], 500.0 / 6.0)
        self.assertEqual(m["dias_total"], 2)
        self.assertEqual(m["pct_dias_com_algum_ativo_ge_n"], 100.0)
        self.assertEqual(m["pct_dias_com_algum_ativo_gt_n"], 50.0)
        self.assertEqual(m["pct_dias_com_total_noticias_ge_n"], 100.0)
        self.assertEqual(m["pct_dias_com_total_noticias_gt_n"], 100.0)
        self.assertEqual(m["pct_dias_com_noticias"], 100.0)

    def test_n_zero_inclusivo_e_estrito(self):
        m = calcular_metricas_cobertura(self.q, 0)
        self.assertEqual(m["pct_elegiveis"], 100.0)
        self.assertEqual(m["pct_ativo_dias_gt_n"], 75.0)
        self.assertEqual(m["pct_noticias_dentro_padrao"], 100.0)

    def test_sem_noticias_percentual_indefinido(self):
        q = self.q.copy()
        q["news_count"] = 0
        m = calcular_metricas_cobertura(q, 1)
        self.assertTrue(pd.isna(m["pct_noticias_dentro_padrao"]))
        self.assertEqual(m["pct_dias_com_noticias"], 0.0)
        self.assertEqual(m["pct_dias_com_algum_ativo_ge_n"], 0.0)

    def test_duplicatas_nao_sao_aceitas(self):
        q = pd.concat([self.q, self.q.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            calcular_metricas_cobertura(q, 2)

    def test_limiar_invalido(self):
        with self.assertRaises(ValueError):
            calcular_metricas_cobertura(self.q, -1)


if __name__ == "__main__":
    unittest.main()
