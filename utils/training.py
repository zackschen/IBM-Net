# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from dataset.utils.continual_dataset import ContinualDataset
from typing import Tuple
from dataset import get_dataset
import sys
from numpy import *
import copy, time

def get_model(model):
    return copy.deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(copy.deepcopy(state_dict))
    return

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

def validation(model: ContinualModel, dataset, task_id) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    total_loss = 0
    correct, total = 0.0, 0.0
    for data in dataset.val_loader:
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device) - task_id * dataset.N_CLASSES_PER_TASK
            outputs = model(inputs, task_id)
            loss = model.loss(outputs, labels)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            total_loss += loss.item()

    model.net.train(status)
    total_loss = total_loss / total
    acc = correct / total * 100
    return total_loss, acc


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs = []
    for k, test_loader in enumerate(dataset.test_loaders):
        model.before_eval(k)
        
        correct, total = 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device) - k * dataset.N_CLASSES_PER_TASK
                outputs = model(inputs, k)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

        accs.append(correct / total * 100)

    model.net.train(status)
    return accs


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace, file) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results = []
    acc_matrix = np.zeros((dataset.N_TASKS,dataset.N_TASKS))

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results = evaluate(model, dataset_copy)

    print(f'Radnom Accuracies =',file=file, flush=True)
    for j_a in range(len(random_results)): 
        print('{:5.1f}% '.format(random_results[j_a]),end='',file=file, flush=True)
    print(file=file, flush=True)
    for t in range(dataset.N_TASKS):
        best_loss=np.inf
        best_model=get_model(model)
        lr = args.lr

        model.current_training_task = t
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)

        scheduler = dataset.get_scheduler(model, args)
        tstart=time.time()
        for epoch in range(model.args.n_epochs):
            model.epoch_start(epoch, t)
            model.current_epoch = epoch
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits, t)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, t = t)

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)
            model.epoch_end(epoch, t)

        print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000), file=file, flush=True)

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs)
        
        for acc_i in range(t+1):
            acc_matrix[t,acc_i] = results[t][acc_i]
        print(file=file, flush=True)
        print(f'Task:{t} Accuracies =',file=file, flush=True)
        average = []
        for i_a in range(t+1):
            print('\t',end='',file=file, flush=True)
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='',file=file, flush=True)
            print(file=file, flush=True)
            average.append(acc_matrix[i_a][:i_a+1].mean())
        print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[t].mean()),file=file, flush=True)
        bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
        print('Backward transfer: {:5.2f}%'.format(bwt),file=file, flush=True)
        print('Mean Avg Accuracy: {:5.2f}%'.format(np.mean(average)),file=file, flush=True)

        mean_acc = np.mean(accs)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING,file=file)

        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    if args.csv_log:
        csv_logger.add_bwt(results)
        csv_logger.add_forgetting(results)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
