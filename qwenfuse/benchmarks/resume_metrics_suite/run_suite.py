"""Missing résumé evidence only. No experiment starts without --execute."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import legacy_suite as legacy
from common import ROOT, dump, sha
from experiment import plan

HERE = Path(__file__).resolve().parent

def inventory(out):
    return {str(p.relative_to(out)): sha(p)
            for folder in ('code','overlays') for p in (out/folder).rglob('*.py')}

def verify_snapshot(out):
    expected=json.loads((out/'source-hashes.json').read_text())
    assert inventory(out)==expected, 'Frozen Python source changed; use a new run'
    legacy.check_binaries(out)

def freeze(out):
    overlays, libs = legacy.freeze(out)
    joint = out/'overlays/JOINT'
    shutil.copytree(Path(overlays['S1'])/'vllm', joint/'vllm',symlinks=True)
    target=joint/'vllm'/legacy.NATIVE
    target.unlink()
    target.symlink_to(libs['C4'])
    legacy.force_four(joint,True)
    overlays['JOINT']=str(joint)
    libs['JOINT']=libs['C4']
    dump(out/'paths.json',dict(overlays=overlays,libraries=libs))
    dump(out/'source-hashes.json',inventory(out))
    return overlays,libs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight',action='store_true',help='Read-only GPU/environment check; no model loading')
    a=p.parse_args();out=a.out.resolve()
    if a.resume:
        frozen=out/'code/run_suite.py'
        if not frozen.is_file(): raise RuntimeError('No frozen run to resume')
        if Path(__file__).resolve()!=frozen.resolve():
            os.execv(sys.executable,[sys.executable,str(frozen),*sys.argv[1:]])
    jobs=plan()
    if not a.execute:
        print(json.dumps(dict(jobs=len(jobs),model_processes=sum(j['kind']=='model' for j in jobs),
                              plan=jobs),indent=2))
        if a.preflight:
            subprocess.run([sys.executable,'-c',
                'import json,legacy_suite; print(json.dumps(legacy_suite.preflight(),indent=2))'],
                cwd=HERE,check=True)
        print('PLAN PASS; no experiment executed. Add --execute to start.')
        return
    ROOT.joinpath('reports').mkdir(parents=True,exist_ok=True)
    with (ROOT/'reports/qwenfuse-formal.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not a.resume: out.mkdir(parents=True,exist_ok=False)
        else:
            verify_snapshot(out)
            assert json.loads((out/'plan.json').read_text())==jobs,'Plan differs'
        def interrupted(s,f): raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,interrupted)
        try:
            envfile=out/('resume-environment-'+str(time.time_ns())+'.json' if a.resume else 'environment.json')
            subprocess.run([sys.executable,'-c',
                'import legacy_suite; from common import dump; dump('+repr(str(envfile))+',legacy_suite.preflight())'],
                cwd=HERE,check=True)
            if a.resume:
                old=json.loads((out/'environment.json').read_text())
                new=json.loads(envfile.read_text())
                for k in ('python','torch','cuda','model_config'):
                    assert old[k]==new[k],f'Environment changed: {k}'
                paths=json.loads((out/'paths.json').read_text())
                overlays,libs=paths['overlays'],paths['libraries']
            else:
                dump(out/'plan.json',jobs)
                shutil.copytree(HERE,out/'code',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
                overlays,libs=freeze(out)
            dump(out/'controller.json',dict(pid=os.getpid(),resumed=a.resume,time=time.time()))
            for j in jobs:
                dest=out/'stages'/j['id']
                if a.resume and (dest/'status.json').exists():
                    status=json.loads((dest/'status.json').read_text())
                    if status.get('state')=='PASS':
                        assert json.loads((dest/'job.json').read_text())==j
                        json.loads((dest/'result.json').read_text())
                        if j.get('check_only'):
                            assert (dest/'logits.npy').is_file() and (dest/'kernel-path.json').is_file()
                        print('SKIP PASS',j['id'],flush=True)
                        continue
                if dest.exists():
                    archive=out/'interrupted-attempts'/str(time.time_ns())/j['id']
                    archive.parent.mkdir(parents=True,exist_ok=True)
                    dest.rename(archive)
                legacy.stage(out,j,overlays,libs)
            verify_snapshot(out)
            subprocess.run([sys.executable,str(out/'code/report.py'),str(out)],check=True)
            dump(out/'status.json',dict(state='COMPLETED_REVIEW_REQUIRED',jobs=len(jobs),
                note='Read numerical review, drift, control regressions; no automatic accuracy claim.'))
            print('FINISHED:',out/'summary.txt',flush=True)
        except BaseException as e:
            dump(out/'status.json',dict(state='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'FAILED',
                                      error=str(e),pid=os.getpid()))
            raise

if __name__=='__main__': main()
