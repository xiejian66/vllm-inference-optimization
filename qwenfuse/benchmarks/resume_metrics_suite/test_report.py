"""Synthetic artifacts verify report math and pairing, without GPU work."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from common import dump
from experiment import plan
from report import numeric

class ReportTests(unittest.TestCase):
    def test_numerical_identity_and_difference(self):
        a=np.array([[1.,2.,3.]])
        self.assertTrue(numeric(a,a)['exact'])
        x=numeric(a,a+.125)
        self.assertFalse(x['exact'])
        self.assertEqual(x['max_abs'],.125)
        self.assertEqual(x['argmax_agreement'],1.)

    def test_full_report_pairing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); jobs=plan(); dump(root/'plan.json',jobs)
            def stats(x): return dict(median=x,min=x,max=x,std=0.,n=3)
            for j in jobs:
                dest=root/'stages'/j['id'];dest.mkdir(parents=True)
                dump(dest/'status.json',dict(state='PASS'))
                if j['purpose']=='numeric':
                    np.save(dest/'logits.npy',np.array([[[1.,2.,3.]]]))
                    dump(dest/'logits-inputs.json',dict(prompts=[[1,2]],steps=1))
                    result={}
                elif j['kind']=='micro':result={'median_us':{'A4':10.,'B4':8.,'C4':6.}[j['variant']]}
                else:
                    secs={'A4':10.,'C4':9.,'S0':10.,'S1':9.,'JOINT':8.}[j['variant']]
                    result=dict(results=[dict(batch=8 if j['purpose']=='anomaly' else 64,
                        workload_sha256='same-inputs',latency=stats(secs),throughput=stats(100./secs))])
                dump(dest/'result.json',result)
            subprocess.run([sys.executable,str(Path(__file__).with_name('report.py')),str(root)],
                           check=True,stdout=subprocess.DEVNULL)
            pairs=json.loads((root/'paired-performance.json').read_text())
            joint=[r for r in pairs if r['baseline']=='S0' and r['candidate']=='JOINT']
            self.assertEqual(len(joint),6)
            for row in joint:
                self.assertAlmostEqual(row['latency_reduction_pct'],20.)
                self.assertAlmostEqual(row['throughput_gain_pct'],25.)
            numeric_results=json.loads((root/'numerical-review.json').read_text())
            self.assertEqual(len(numeric_results),11)
            self.assertTrue(all(x['exact'] for x in numeric_results.values()))

if __name__=='__main__':unittest.main()
