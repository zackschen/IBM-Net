# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import sys
from typing import Dict, Any
from utils.metrics import *

from utils import create_if_not_exists
from utils.conf import base_path
import numpy as np

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str,file) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    print('\nAccuracy for {} task(s): \t  [Task-IL]: {} %\n'.format(task_number, round(mean_acc, 2)),file=file, flush=True)


class CsvLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        self.accs = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

    def add_fwt(self, results, accs):
        self.fwt = forward_transfer(results, accs)

    def add_bwt(self, results):
        self.bwt = backward_transfer(results)

    def add_forgetting(self, results):
        self.forgetting = forgetting(results)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        else:
            self.accs.append(mean_acc)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        for cc in useless_args:
            if cc in args:
                del args[cc]

        columns = list(args.keys())

        new_cols = []
        for i, acc in enumerate(self.accs):
            args['task' + str(i + 1)] = acc
            new_cols.append('task' + str(i + 1))

        args['forward_transfer'] = self.fwt
        new_cols.append('forward_transfer')

        args['backward_transfer'] = self.bwt
        new_cols.append('backward_transfer')

        args['forgetting'] = self.forgetting
        new_cols.append('forgetting')

        columns = new_cols + columns

        create_if_not_exists(base_path() + "results/" + self.setting)
        create_if_not_exists(base_path() + "results/" + self.setting +
                             "/" + self.dataset)
        create_if_not_exists(base_path() + "results/" + self.setting +
                             "/" + self.dataset + "/" + self.model)

        write_headers = False
        path = base_path() + "results/" + self.setting + "/" + self.dataset\
               + "/" + self.model + "/mean_accs.csv"
        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)
