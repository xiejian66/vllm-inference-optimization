import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class PortableTests(unittest.TestCase):
    def test_paths_follow_environment_even_from_another_cwd(self):
        with tempfile.TemporaryDirectory(prefix='qwenfuse path ') as d:
            env=os.environ.copy();env.update(QWENFUSE_WORKSPACE=d,QWENFUSE_MODEL=d+'/weights',PYTHONPATH=str(Path(__file__).parent))
            output=subprocess.check_output([sys.executable,'-c',"import json;from runtime_config import ROOT,MODEL,REPOS;print(json.dumps([str(ROOT),str(MODEL),str(REPOS['A4']),str(REPOS['S1'])]))"],env=env,cwd='/tmp',text=True)
            self.assertEqual(json.loads(output),[d,d+'/weights',d+'/src/A',d+'/src/D'])
    def test_stable_micro_is_formal_default(self):
        from experiment import plan
        jobs=[j for j in plan() if j['kind']=='micro']
        self.assertEqual({j['tokens'] for j in jobs},{1,8,32,64,128,256})
        self.assertTrue(all(j['samples']==100 and j['unroll']==128 for j in jobs))
        code=(Path(__file__).parent/'kernel_worker.py').read_text()
        self.assertIn('from stable_micro import measured',code)

if __name__=='__main__':unittest.main()
