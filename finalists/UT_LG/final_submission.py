import argparse
import time
import torch
from utils.io import load_yaml
from types import SimpleNamespace
from utils.names_match_torch import methods
import os
from utils.common import create_code_snapshot
import numpy as np

def main(args):
    params = load_yaml(args.parameters)
    criterion = torch.nn.CrossEntropyLoss()
    params['verbose'] = args.verbose
    print(params, end="\n\n")
    final_params = SimpleNamespace(**params)
    time_start = time.time()
    method = methods[final_params.method](final_params, criterion, final_params.use_cuda)
    valid_acc, elapsed, ram_usage, ext_mem_sz, preds = method.train_model(tune=False)

    # directory with the code snapshot to generate the results
    sub_dir = 'submissions/' + final_params.sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(valid_acc), elapsed, np.average(ram_usage),
            np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')
    parser.add_argument('--parameters', dest='parameters', default='config/final/default.yml')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='print information or not')
    args = parser.parse_args()
    main(args)