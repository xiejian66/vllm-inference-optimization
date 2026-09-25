import json
import sys
from pathlib import Path
import base_model_worker
from experiment import make_workload

if __name__ == '__main__':
    job = json.loads(Path(sys.argv[sys.argv.index('--job') + 1]).read_text())
    def selected_workload(scenario, batch=32, input_len=512, output_len=128):
        return make_workload(scenario, batch, input_len, output_len, seed=job['seed'])
    base_model_worker.workload = selected_workload
    base_model_worker.main()
