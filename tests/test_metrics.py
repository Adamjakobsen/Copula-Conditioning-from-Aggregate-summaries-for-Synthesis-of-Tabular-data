import unittest
import numpy as np
import pandas as pd

from cocast import schema
from cocast.metrics import categorical_tables, fidelity, proximity


def table(n=40):
    rng = np.random.default_rng(9)
    data = {schema.AGE_COL: rng.integers(18, 40, n).astype(float), schema.SEX_COL: rng.integers(1, 3, n)}
    for disorder, spec in schema.DISORDERS.items():
        data.update({key: rng.integers(0, spec['item_max'] + 1, n) for key in schema.item_columns(disorder)})
    return pd.DataFrame(data)


class MetricTests(unittest.TestCase):
    def test_dependence_error_with_identical_marginals(self):
        real = table(400)
        real.loc[:, schema.ALL_ITEM_COLUMNS] = 0
        a, b = schema.ALL_ITEM_COLUMNS[:2]
        real[a] = np.tile([0, 0, 1, 1], 100)
        real[b] = real[a]
        synth = real.copy()
        synth[b] = 1 - real[a]
        result = fidelity(real, synth)
        self.assertAlmostEqual(result['jsd'], 0)
        self.assertGreater(result['energy'], .01)

    def test_identical_table_has_zero_fidelity(self):
        data = table()
        for value in fidelity(data, data).values():
            self.assertAlmostEqual(value, 0)

    def test_real_self_matches_excluded_but_duplicates_remain(self):
        real = table()
        self.assertEqual(proximity(real, real)['exact_percent'], 100)
        self.assertEqual(proximity(real)['exact_percent'], 0)
        real.iloc[1] = real.iloc[0]
        self.assertEqual(proximity(real)['exact_percent'], 5)

    def test_missing_age_is_distinct_from_age_bin(self):
        real = table()
        real.loc[0, schema.AGE_COL] = np.nan
        x, y = categorical_tables(real, real)
        self.assertEqual(x[0, 0], y[0, 0])
        self.assertNotIn(x[0, 0], x[1:, 0])


if __name__ == '__main__':
    unittest.main()
