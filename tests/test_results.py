"""Check paired differences, uncertainty sources and reference deduplication."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from cocast.io import sha256, write_json
from cocast.results import summarize


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def evaluation(self, seed):
        path = self.root / str(seed)
        path.mkdir()
        rows = []
        for repeat in (1, 2, 3):
            for condition, offset in [('natural',0),('targeted',.1)]:
                rows.append(dict(run=condition,model='Qwen/test',generation_seed=seed,context='own',
                    regime=condition, family='utility_repeat', domain='all_domains',metric='f1_macro',
                    protocol='TAUG',classifier='histgb',repeat=repeat,value=repeat*.1+seed*.01+offset))
            rows.append(dict(run='real',family='utility_repeat',domain='all_domains',metric='f1_macro',
                             classifier='histgb',protocol='TRTR',repeat=repeat,value=repeat*.1))
        pd.DataFrame(rows).to_csv(path/'metrics.csv',index=False)
        write_json(path/'manifest.json',dict(status='complete',real_sha256='same',
            utility={'protocol_version':'fixed'},files={'metrics.csv':sha256(path/'metrics.csv')}))
        return path

    def test_seed_average_then_repeat_sd_and_paired_subtraction(self):
        paths = [self.evaluation(i) for i in (1,2,3)]
        summarize(paths, self.root/'results')
        rows = pd.read_csv(self.root/'results/summary.csv')
        natural = rows[rows.condition.eq('natural:own')].iloc[0]
        self.assertAlmostEqual(natural['mean'], .22)
        self.assertAlmostEqual(natural['sd'], .1)
        self.assertEqual(natural.n_generation_seeds,3)
        real = rows[rows.model.eq('real')].iloc[0]
        self.assertEqual(real.n_repeats,3)
        self.assertAlmostEqual(real['sd'],.1)
        paired = pd.read_csv(self.root/'results/targeting_changes.csv').iloc[0]
        self.assertAlmostEqual(paired['mean'],.1)
        self.assertAlmostEqual(paired['sd'],0)

    def test_tampered_evaluation_and_duplicate_input_rejected(self):
        path = self.evaluation(1)
        with self.assertRaisesRegex(ValueError,'once'):
            summarize([path,path], self.root/'results')
        with (path/'metrics.csv').open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError,'checksum'):
            summarize([path],self.root/'results')
