"""Package the existing experimental risk scheduler without changing its policy."""
import ast
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parents[2]
helpers = root / 'qwenfuse/benchmarks/resume_metrics_suite'
target = root / 'vllm/v1/core/sched'
scheduler = target / 'scheduler.py'

def main():
    # Extract pure prediction helpers; no diagnostic scheduler dependency is needed.
    text = (helpers / 'risk_scheduler.py').read_text()
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict_fresh')
    predictor = ('from dataclasses import asdict\n'
                 'from vllm.v1.request import RequestStatus\n'
                 'from .qwenfuse_prompt_risk import inspect_risk, full_prompt_new_blocks\n\n'
                 + ast.get_source_segment(text, fn) + '\n')
    policy = (helpers / 'risk_perf_scheduler.py').read_text().replace(
        'from risk_scheduler import predict_fresh',
        'from .qwenfuse_risk_predict import predict_fresh')
    files = {'qwenfuse_risk_predict.py': predictor,
             'qwenfuse_risk.py': policy,
             'qwenfuse_prompt_risk.py': (helpers / 'prompt_risk_v1.py').read_text()}
    for name, content in files.items():
        ast.parse(content)
        existing = target / name
        if existing.exists() and existing.read_text() != content:
            raise RuntimeError(f'Refusing to overwrite changed file: {existing}')
    spec = importlib.util.spec_from_file_location('patch', helpers / 'patch_risk_policy.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if 'risk_reorder_checked = False' not in scheduler.read_text():
        module.patch_scheduler(scheduler)
    for name, content in files.items():
        (target / name).write_text(content)
    print('PASS: risk scheduler source prepared; no build or inference executed.')

if __name__ == '__main__':
    main()
