import argparse
import json
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from paccmann_sets.models.set_matching.rnn import RNNSetMatching
from paccmann_sets.utils.hyperparameters import (
    DISTRIBUTION_FUNCTION_FACTORY, LR_SCHEDULER_FACTORY
)
from paccmann_sets.utils.loss_setmatching import SetMatchLoss
from pytoda.datasets.distributional_dataset import DistributionalDataset
from pytoda.datasets.set_matching_dataset import (
    PairedSetMatchingDataset, PermutedSetMatchingDataset
)
from pytoda.datasets.utils.factories import METRIC_FUNCTION_FACTORY
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, help='Path to save best model.')
parser.add_argument('results_path', type=str, help='Path to save results and plots.')
parser.add_argument(
    'training_params', type=str, help='Path to the training parameters.'
)


def main(model_path: str, results_path: str, training_params: str):
    '''Performs a comparative analysis of set matching networks with different
        recurrent cells.

    Args:
        model_path (str): Path where the best model should be saved.
        results_path (str): Path where all the results and plots should be saved.
        training_params (str): Path where the training parameters of the model
            are contained.
    '''

    torch.Tensor.ndim = property(lambda self: len(self.shape))
    # setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(results_path, 'set_matchingRNN.log')),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger('sets')
    logger.setLevel(logging.DEBUG)

    # setup plots
    clrs = sns.color_palette('Paired', 8)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title(
        'Train and Validation Loss of Various Recurrent Cells on Set Matching Task'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cross Entropy (Row) Loss')

    fig2, (ax2, ax3) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    ax2.set_title(
        'Train and Validation Accuracy of Various Recurrent Cells on Set Matching Task'
    )
    ax2.set(ylabel='Average per Set Accuracy')
    ax3.set(xlabel='Epochs', ylabel='Proportion of Samples with 100% Accuracy')

    dataset_dict = {
        'permuted': PermutedSetMatchingDataset,
        'sampled': PairedSetMatchingDataset,
    }

    with open(training_params, 'r') as readjson:
        params = json.load(readjson)

    train_size = params.get('train_size', 20000)
    valid_size = params.get('valid_size', 5000)
    test_size = params.get('test_size', 20000)

    input_dim = params.get('input_size', 8)
    max_length = params.get('max_length', 5)
    min_length = params.get('min_length', 2)
    padding_value = params.get('padding_value', 4.0)
    noise_std = params.get('noise_std', 0.0)

    dist_type = params.get('distribution_type', 'normal')
    dist_args = params.get('distribution_args', {'loc': 0.0, 'scale': 1.0})
    tensor_args = {}
    for key, value in dist_args.items():
        if key == 'covariance_matrix':
            value = torch.tensor(value, device=device)
            identity_matrix = torch.eye(len(value), device=device)
            identity_matrix[range(len(value)), range(len(value))] = value
            tensor_args[key] = identity_matrix
        else:
            tensor_args[key] = torch.tensor(value, device=device)

    metric = params.get('cost_metric', 'p-norm')
    metric_args = params.get('cost_metric_args', {'p': 2})

    seeds = params['seeds']
    for k, i in seeds.items():
        seeds[k] = eval(i)

    dataset_type = params.get('dataset_type', 'permute')

    batch_first = eval(params['batch_first'])

    dist_function = DISTRIBUTION_FUNCTION_FACTORY[dist_type](**tensor_args)
    cost_function = METRIC_FUNCTION_FACTORY[metric](**metric_args)

    # setup datasets and dataloader
    train, valid, test = [], [], []

    dataset_train_1 = DistributionalDataset(
        train_size, (max_length, input_dim), dist_function, seed=seeds['train1']
    )

    dataset_valid_1 = DistributionalDataset(
        valid_size, (max_length, input_dim), dist_function, seed=seeds['valid1']
    )

    dataset_test_1 = DistributionalDataset(
        test_size, (max_length, input_dim), dist_function, seed=seeds['test1']
    )

    train.append(dataset_train_1)
    valid.append(dataset_valid_1)
    test.append(dataset_test_1)

    if dataset_type == 'sampled':
        dataset_train_2 = DistributionalDataset(
            train_size, (max_length, input_dim), dist_function, seed=seeds['train2']
        )
        dataset_valid_2 = DistributionalDataset(
            valid_size, (max_length, input_dim), dist_function, seed=seeds['valid2']
        )
        dataset_test_2 = DistributionalDataset(
            test_size, (max_length, input_dim), dist_function, seed=seeds['test2']
        )
        train.append(dataset_train_2)
        valid.append(dataset_valid_2)
        test.append(dataset_test_2)

    dataset_train = dataset_dict[dataset_type](
        *train,
        min_length,
        cost_function,
        padding_value,
        noise_std,
        seed=seeds['train_true'],
    )

    dataset_test = dataset_dict[dataset_type](
        *test,
        min_length,
        cost_function,
        padding_value,
        noise_std,
        seed=seeds['test_true'],
    )

    dataset_valid = dataset_dict[dataset_type](
        *valid,
        min_length,
        cost_function,
        padding_value,
        noise_std,
        seed=seeds['valid_true'],
    )

    train_loader = DataLoader(
        dataset_train, batch_size=params['train_batch'], shuffle=True
    )

    valid_loader = DataLoader(
        dataset_valid, batch_size=params['valid_batch'], shuffle=True
    )

    test_loader = DataLoader(
        dataset_test, batch_size=params['test_batch'], shuffle=True
    )

    epochs = params['epochs']

    connector = torch.zeros(1, input_dim).fill_(99.0)

    clr_idx = 0

    patience = params.get('patience', 5)
    epochs_no_change = 0

    for cell in ['GRU', 'nBRC', 'LSTM', 'BRC']:

        params.update({'cell': cell})

        # setup model params
        save_here = os.path.join(model_path, '{}_{}'.format(params['loss'], cell))
        model = None
        if model is not None:
            del model

        # setup model and optimiser
        model = RNNSetMatching(params, device).to(device)

        optimzr = torch.optim.Adam(model.parameters(), lr=params['lr'])
        lr_scheduler = LR_SCHEDULER_FACTORY[params['scheduler']
                                            ](optimzr, **params['lr_args'])

        min_loss = np.inf

        loss_fn = SetMatchLoss(
            params['loss'], params['ce_type'], params['temperature'],
            params['sinkhorn_iter']
        )

        plot_train = []
        plot_valid = []
        set_acc_train = []
        set_acc_valid = []
        acc100_train = []
        acc100_valid = []
        epoch_times = []
        model_times = []

        for epoch in range(epochs):

            time_train = 0
            t0 = time.time()

            model.train()
            logger.info('=== {} Epoch [{}/{}]'.format(cell, epoch + 1, epochs))

            for (
                idx,
                (x1_train, x2_train, target12_train, target21_train, lens_train),
            ) in enumerate(train_loader):

                bs_train = len(target12_train)
                connector_ = connector.expand(bs_train, -1, -1).to(device)

                train_x = torch.cat((x1_train, connector_, x2_train), dim=1).to(device)

                if batch_first is False:
                    # permute train_x so batch comes second
                    train_x = train_x.permute(1, 0, 2)

                target12_train, target21_train = (
                    target12_train.to(device),
                    target21_train.to(device),
                )

                tic = time.time()
                train_output = model(train_x)
                torch.cuda.current_stream().synchronize()
                toc = time.time()

                time_train += toc - tic

                if batch_first is False:
                    # permute output so batch comes first for loss calculation
                    train_output = train_output.permute(1, 0, 2)

                train_loss = loss_fn(
                    train_output, target12_train, target21_train, lens_train
                )
                optimzr.zero_grad()
                train_loss.backward()
                optimzr.step()

                if idx == (len(train_loader) - 2):

                    # Replace 2 with 1 and target21_train with target12_train
                    # for column-wise objective functions such as KL-Div_col.
                    train_preds = torch.argmax(train_output.detach(), 2)
                    all_correct = (train_preds == target21_train).all(dim=1
                                                                      ).float().sum()
                    correct_train = (train_preds == target21_train).float().sum()

                    plot_train.append(train_loss.detach())
                    logger.info('Train Loss = {}'.format(train_loss.detach()))

                    batch_avg_acc = correct_train / (max_length * bs_train)
                    set_acc_train.append(batch_avg_acc)
                    logger.info('Train per set Acc = {}'.format(batch_avg_acc))

                    acc_train = all_correct / bs_train
                    acc100_train.append(acc_train)
                    logger.info('Train 100% Acc = {}'.format(acc_train))

            t1 = time.time()
            epoch_times.append(t1 - t0)
            model_times.append(time_train)

            model.eval()
            avg_valid_loss = 0
            accuracy_valid = []
            all_correct_valid = 0

            for (
                idx,
                (x1_valid, x2_valid, target12_valid, target21_valid, lens_valid),
            ) in enumerate(valid_loader):

                bs_valid = len(target12_valid)
                connector_ = connector.expand(bs_valid, -1, -1).to(device)

                valid_x = torch.cat((x1_valid, connector_, x2_valid), dim=1).to(device)

                if batch_first is False:
                    # permute valid_x so batch comes second
                    valid_x = valid_x.permute(1, 0, 2)

                target12_valid, target21_valid = (
                    target12_valid.to(device),
                    target21_valid.to(device),
                )

                valid_output = model(valid_x)

                if batch_first is False:
                    # permute output so batch comes first for loss calculation
                    valid_output = valid_output.permute(1, 0, 2)

                valid_loss = loss_fn(
                    valid_output, target12_valid, target21_valid, lens_valid
                )

                avg_valid_loss = (avg_valid_loss * idx +
                                  valid_loss.detach()) / (idx + 1)

                # Replace 2 with 1 and target21_valid with target12_valid
                # for column-wise objective functions such as KL-Div_col.
                valid_preds = torch.argmax(valid_output.detach(), 2)
                all_correct_valid += (
                    (valid_preds == target21_valid).all(dim=1).float().sum()
                )
                correct_valid = (valid_preds == target21_valid).float().sum()

                accuracy_valid.append(correct_valid / (max_length * bs_valid))

            plot_valid.append(avg_valid_loss.detach())
            logger.info('Avg Valid Loss = {}'.format(avg_valid_loss.detach()))

            avg_valid_acc = torch.mean(torch.as_tensor(accuracy_valid))
            set_acc_valid.append(avg_valid_acc)
            logger.info('Valid per set Acc = {}'.format(avg_valid_acc))

            acc_valid = all_correct_valid / valid_size
            acc100_valid.append(acc_valid)
            logger.info('Valid 100% Acc = {}'.format(acc_valid))

            if avg_valid_loss < min_loss:
                min_loss = avg_valid_loss
                epochs_no_change = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimzr.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss,
                    },
                    save_here,
                )
            else:
                if epoch >= 19:
                    epochs_no_change += 1

            if epochs_no_change == patience:
                logger.info('Early Stopping at {} epochs'.format(epoch))
                break

            lr_scheduler.step(avg_valid_loss.detach())

        logger.info(
            'Average processing time for {}: {} seconds'.format(
                cell, np.mean(model_times)
            )
        )
        logger.info(
            'Total epochs time for {} : {} hours'.format(cell,
                                                         sum(epoch_times) / 3600)
        )

        torch.save(model_times, os.path.join(results_path, f'model_times_{cell}'))
        torch.save(epoch_times, os.path.join(results_path, f'epoch_times_{cell}'))

        torch.save(plot_train, os.path.join(results_path, 'train_loss_{}'.format(cell)))
        torch.save(
            plot_valid, os.path.join(results_path, 'avg_valid_loss_{}'.format(cell))
        )
        ax.plot(
            range(len(plot_train)),
            plot_train,
            color=clrs[clr_idx * 2],
            label='{} Training Loss'.format(cell)
        )
        ax.plot(
            range(len(plot_valid)),
            plot_valid,
            color=clrs[clr_idx * 2 + 1],
            label='{} Validation Loss'.format(cell)
        )

        ax2.plot(
            range(len(set_acc_train)),
            set_acc_train,
            color=clrs[clr_idx * 2],
            label='{} Batch Training Accuracy'.format(cell),
        )
        ax2.plot(
            range(len(set_acc_valid)),
            set_acc_valid,
            color=clrs[clr_idx * 2 + 1],
            label='{} Validation Accuracy'.format(cell),
        )

        ax3.plot(
            range(len(acc100_train)),
            acc100_train,
            color=clrs[clr_idx * 2],
            label='{} Batch Training Accuracy'.format(cell),
        )
        ax3.plot(
            range(len(acc100_valid)),
            acc100_valid,
            color=clrs[clr_idx * 2 + 1],
            label='{} Validation Accuracy'.format(cell),
        )

        avg_test_loss = 0
        all_correct_test = 0
        model.eval()

        test_predlist = []
        test_truelist12 = []
        test_truelist21 = []
        accuracy_test = []
        x1_list = []
        x2_list = []
        test_lengths = []

        for idx, (x1_test, x2_test, target12_test, target21_test,
                  lens_test) in enumerate(test_loader):

            bs_test = len(target12_test)
            connector_ = connector.expand(bs_test, -1, -1).to(device)
            test_x = torch.cat((x1_test, connector_, x2_test), dim=1).to(device)

            if batch_first is False:
                # permute test_x so batch comes second
                test_x = test_x.permute(1, 0, 2)

            target12_test, target21_test = (
                target12_test.to(device),
                target21_test.to(device),
            )

            test_output = model(test_x)

            if batch_first is False:
                # permute output so batch comes first for loss calculation
                test_output = test_output.permute(1, 0, 2)

            test_loss = loss_fn(test_output, target12_test, target21_test, lens_test)

            test_predlist.append(test_output.detach())
            test_truelist12.append(target12_test.detach())
            test_truelist21.append(target21_test.detach())
            x1_list.append(x1_test.detach())
            x2_list.append(x2_test.detach())
            test_lengths.append(lens_test)

            # Replace 2 with 1 and target21_valid with target12_valid
            # for column-wise objective functions such as KL-Div_col.

            test_preds = torch.argmax(test_output, 2)
            all_correct_test += (test_preds == target21_test).all(dim=1).float().sum()
            correct_test = (test_preds == target21_test).float().sum()

            accuracy_test.append(correct_test / (max_length * bs_test))

            avg_test_loss = (avg_test_loss * idx + test_loss.detach()) / (idx + 1)

        logger.info('Avg Test Loss = {}'.format(avg_test_loss.detach()))
        logger.info(
            'Avg Test per set Acc = {}'.format(
                torch.mean(torch.as_tensor(accuracy_test))
            )
        )
        logger.info('Avg Test Acc = {}'.format(all_correct_test / test_size))

        torch.save(
            test_truelist12, os.path.join(results_path, 'true_match12_{}'.format(cell))
        )
        torch.save(
            test_truelist21, os.path.join(results_path, 'true_match21_{}'.format(cell))
        )
        torch.save(
            test_predlist, os.path.join(results_path, 'pred_match_{}'.format(cell))
        )
        torch.save(x1_list, os.path.join(results_path, 'set1_{}'.format(cell)))
        torch.save(x2_list, os.path.join(results_path, 'set2_{}'.format(cell)))
        torch.save(
            test_lengths, os.path.join(results_path, 'test_lengths_{}'.format(cell))
        )

        clr_idx += 1

    ax.legend()
    fig.savefig(
        os.path.join(results_path, 'setmatchloss_{}.png'.format(params['loss'])),
        bbox_inches='tight'
    )
    ax2.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    fig2.savefig(
        os.path.join(results_path, 'setmatchacc_{}.png'.format(params['loss'])),
        bbox_inches='tight'
    )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.model_path, args.results_path, args.training_params)
