"""CPU-only release checks; never imports torch/vLLM or starts compilation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[2]
q=root/'qwenfuse'

def main():
    paths=list((q/'benchmarks').rglob('*.py'))+list((q/'scripts').rglob('*.py'))
    for p in paths:ast.parse(p.read_text(),filename=str(p))
    for p in (q/'patches').glob('*.patch'):
        subprocess.run(['git','apply','--numstat',str(p)],check=True,stdout=subprocess.DEVNULL)
    forbidden=('/root/data/qwenfuse','/home/xj/','instance-jxa3hkmb','instance-807zr01i')
    for p in [*paths,*list((q/'benchmarks').rglob('*.sh'))]:
        if p.name=='check_package.py':continue
        assert not any(x in p.read_text() for x in forbidden),f'Old execution path: {p}'
    for p in (q/'benchmarks').rglob('*.sh'):
        subprocess.run(['bash','-n',str(p)],check=True)
    suite=q/'benchmarks/resume_metrics_suite'
    subprocess.run([sys.executable,'-m','unittest','test_plan','test_report','test_portable'],cwd=suite,check=True)
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable,str(q/'scripts/prepare_variants.py'),'--workspace',str(Path(d)/'not-created')],check=True,stdout=subprocess.DEVNULL)
        assert not (Path(d)/'not-created').exists()
        subprocess.run([sys.executable,str(suite/'run_suite.py'),'--out',str(Path(d)/'not-run')],check=True,stdout=subprocess.DEVNULL)
        assert not (Path(d)/'not-run').exists()
    print(f'PASS: {len(paths)} Python files, patch/shell checks, unit tests and no-execution plans.')
if __name__=='__main__':main()
