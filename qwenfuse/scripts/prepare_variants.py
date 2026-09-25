"""Recreate independent ablation trees; default is a plan without network or writes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

BASE='27757dde020ecda4f9b0e2c5ca2df1c29badc82f'
PACKAGE=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path(os.environ.get('QWENFUSE_WORKSPACE','.qwenfuse-work')))
    p.add_argument('--base-repo',type=Path,help='Optional local Git repository containing the fixed base commit')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(); work=a.workspace.expanduser().resolve(); upstream=work/'upstream'
    targets={v:work/'src'/v for v in 'ABCD'}
    patches={v:PACKAGE/'qwenfuse/patches'/f'base-to-{v}.patch' for v in 'BCD'}
    for f in patches.values():
        if not f.is_file():p.error(f'Missing patch: {f}')
    print(json.dumps({'base':BASE,'workspace':str(work),'targets':{k:str(v) for k,v in targets.items()}},indent=2))
    if not a.execute:
        print('PLAN ONLY: no fetch, worktree creation or compilation.');return
    for path in [upstream,*targets.values()]:
        if path.exists():p.error(f'Refusing to overwrite {path}; choose a fresh workspace')
    if a.base_repo:
        source=a.base_repo.expanduser().resolve()
        subprocess.run(['git','-C',str(source),'cat-file','-e',BASE+'^{commit}'],check=True)
    work.mkdir(parents=True,exist_ok=True)
    def run(*args):subprocess.run(args,check=True)
    run('git','init',str(upstream))
    if a.base_repo:run('git','-C',str(upstream),'fetch',str(source),BASE)
    else:run('git','-C',str(upstream),'fetch','--depth=1','https://github.com/vllm-project/vllm.git',BASE)
    for v,dest in targets.items():
        run('git','-C',str(upstream),'worktree','add','--detach',str(dest),BASE)
        if v!='A':
            run('git','-C',str(dest),'apply','--check',str(patches[v]))
            run('git','-C',str(dest),'apply',str(patches[v]))
    (work/'variants.json').write_text(json.dumps({'base':BASE,'variants':{k:str(v) for k,v in targets.items()}},indent=2))
    print('SOURCE READY. No dependency install, compilation or GPU operation performed.')

if __name__=='__main__':main()
