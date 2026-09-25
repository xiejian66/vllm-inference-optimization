"""Link compiled C native and A auxiliary extensions into Python-only D."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main',type=Path,required=True)
    p.add_argument('--aux',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();native='_C_stable_libtorch.abi3.so'
    src=a.main.resolve()/'vllm'/native;aux=a.aux.resolve()/'vllm';dst=a.target.resolve()/'vllm'
    if not dst.is_dir():p.error('Target source does not exist')
    if not src.is_file():p.error('Build C before linking its native extension')
    pairs=[(src,dst/native)]+[(x.resolve(),dst/x.relative_to(aux)) for x in aux.rglob('*.so') if x.name!=native]
    for source,target in pairs:
        if target.exists() or target.is_symlink():
            if target.resolve()==source:continue
            p.error(f'Refusing to overwrite {target}')
        print(f'{target} -> {source}')
    if a.execute:
        for source,target in pairs:
            if target.exists() or target.is_symlink():continue
            target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source)
    print('LINKED' if a.execute else 'PLAN ONLY')
if __name__=='__main__':main()
