import argparse
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from paccmann_sets.datasets.torch_dataset import Collate, SetsDataset
from paccmann_sets.models.sets_autoencoder import SetsAE
from paccmann_sets.utils.helper import (cpuStats, get_gpu_memory_map, setup_logger)
from paccmann_sets.utils.hyperparameters import LR_SCHEDULER_FACTORY
from paccmann_sets.utils.loss_setae import SetAELoss
from paccmann_sets.utils.mapper import MapperSetsAE
from paccmann_sets.utils.setsae_setm import NetworkMapperSetsAE
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    'model_path',
    type=str,
    help='Path where the pre-trained matching network is saved.'
)
parser.add_argument(
    'results_path', type=str, help='Path to save the results, logs and best model.'
)

parser.add_argument(
    'training_params', type=str, help='Path to the training parameters json file.'
)

parser.add_argument(
    'matching_params',
    type=str,
    help='Path to the matching network parameters json file.'
)

parser.add_argument('train_data_path', type=str, help='Path to the training data.')
parser.add_argument('valid_data_path', type=str, help='Path to the validation data.')
parser.add_argument('test_data_path', type=str, help='Path to the testing data.')

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    model_path: str,
    results_path: str,
    training_params: str,
    matching_params: str,
    train_data_path: str,
    valid_data_path: str,
    test_data_path: str,
):
    """Executes the Set Autoencoder for the chosen matching method in
        reconstructing the given data.

    Args:
        model_path (str): Path where the pre-trained matching network is saved
        results_path (str): Path to save the results, logs and best model.
        training_params (str): Path to the training parameters json file.
        matching_params (str): Path to the matching network parameters json file.
        train_data_path (str): Path to the training data.
        valid_data_path (str): Path to the validation data.
        test_data_path (str): Path to the testing data.
    """

    # setup logging
    logger = setup_logger(
        'sets', os.path.join(results_path, "setsae_setm.log"), logging.DEBUG
    )
    logger_mem = setup_logger(
        'memory', os.path.join(results_path, 'logging_memory_time.log')
    )

    clrs = sns.color_palette('Set2', 2)
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle(
        'Train and Validation Loss of Sets AE on Latent Embeddings of GEP and Proteins'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    save_here = os.path.join(results_path, 'setsae')
    model = None
    if model is not None:
        del (model)

    with open(training_params, 'r') as readjson:
        train_params = json.load(readjson)

    with open(matching_params, 'r') as readjson:
        match_params = json.load(readjson)

    max_length = train_params.get('max_length', 5)
    dim = train_params.get('input_size', 128)
    padding_value = train_params.get('padding_value', 4.0)

    train_dataset = SetsDataset(train_data_path, device)
    valid_dataset = SetsDataset(valid_data_path, device)
    test_dataset = SetsDataset(test_data_path, device)

    collator = Collate(max_length, dim, padding_value, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params['train_batch'],
        shuffle=True,
        collate_fn=collator
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=train_params['valid_batch'],
        shuffle=True,
        collate_fn=collator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_params['test_batch'],
        shuffle=False,
        collate_fn=collator
    )

    model = SetsAE(device, **train_params).to(device)

    mapper = NetworkMapperSetsAE(
        train_params['mapper'], model_path, match_params, device
    )
    #mapper =  MapperSetsAE(train_params['matcher'], train_params['p'], device)
    optimiser = torch.optim.Adam(model.parameters(), lr=train_params['lr'])

    lr_scheduler = LR_SCHEDULER_FACTORY[train_params['scheduler']
                                        ](optimiser, *train_params['lr_args'])

    loss = SetAELoss(train_params['loss'], device)
    min_loss = np.inf
    epochs = train_params['epochs']

    plot_train = []
    plot_valid = []
    for epoch in range(epochs):

        model.train()
        logger.info("=== Epoch [{}/{}]".format(epoch + 1, epochs))

        if epoch == 1:
            tic = time.time()

        train_time = 0
        for idx, (x_train, train_lengths) in enumerate(train_loader):

            x_train = x_train.to(device)

            pred_train, prob_train = model(x_train, x_train.size(1), train_lengths)

            t0 = time.time()
            mapped_outputs_train, mapped_prob_train, train12 = mapper(
                x_train, pred_train, prob_train
            )
            torch.cuda.current_stream().synchronize()
            t1 = time.time()

            if epoch == 0 and idx == 0:
                logger_mem.info(cpuStats())
                logger_mem.info(print(get_gpu_memory_map()))
                logger_mem.info('x_train:{}'.format(x_train.size()))
                logger_mem.info('pred_train:{}'.format(pred_train.size()))
                logger_mem.info('prob_train:{}'.format(prob_train.size()))
                logger_mem.info('train_lengths:{}'.format(train_lengths.size()))
                logger_mem.info(torch.cuda.memory_allocated())

            train_time += t1 - t0

            train_loss = loss(
                x_train, mapped_outputs_train, mapped_prob_train, train_lengths
            )

            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()

        if epoch == 1:
            toc = time.time()
            logger_mem.info("Total training time for one epoch = {}".format(toc - tic))

        logger_mem.info(
            "Total mapping time in training epoch {}  = {}".format(epoch, train_time)
        )
        plot_train.append(train_loss.detach())
        logger.info("Train Loss = {}".format(train_loss.detach()))

        model.eval()
        avg_valid_loss = 0
        for idx, (x_valid, valid_lengths) in enumerate(valid_loader):

            x_valid = x_valid.to(device)

            pred_valid, prob_valid = model(x_valid, x_valid.size(1), valid_lengths)

            mapped_outputs_valid, mapped_prob_valid, valid12 = mapper(
                x_valid, pred_valid, prob_valid
            )

            valid_loss = loss(
                x_valid, mapped_outputs_valid, mapped_prob_valid, valid_lengths
            )

            avg_valid_loss = (avg_valid_loss * idx + valid_loss.detach()) / (idx + 1)

        plot_valid.append(avg_valid_loss.detach())
        logger.info("Avg Valid Loss = {}".format(avg_valid_loss.detach()))
        if avg_valid_loss < min_loss:
            min_loss = avg_valid_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss
                    # 'learning_rate': lr_expscheduler.get_last_lr()[0]
                },
                save_here
            )
        lr_scheduler.step()

    torch.save(plot_train, os.path.join(results_path, 'train_loss'))
    torch.save(plot_valid, os.path.join(results_path, 'avg_valid_loss'))

    ax.plot(range(len(plot_train)), plot_train, color=clrs[0], label="Training Loss")
    ax.plot(range(len(plot_valid)), plot_valid, color=clrs[1], label="Validation Loss")
    ax.legend()
    fig.savefig(os.path.join(results_path, 'setsAE.png'))

    avg_test_loss = 0
    model.eval()
    test_predlist = []
    test_truelist = []
    test12_list = []
    mapping_time = 0

    for idx, (x_test, test_lengths) in enumerate(test_loader):

        x_test = x_test.to(device)

        pred_test, prob_test = model(x_test, x_test.size(1), test_lengths)

        tic = time.time()
        mapped_outputs_test, mapped_prob_test, test12 = mapper(
            x_test, pred_test, prob_test
        )
        toc = time.time()
        mapping_time += toc - tic

        test_loss = loss(x_test, mapped_outputs_test, mapped_prob_test, test_lengths)

        test_predlist.append(mapped_outputs_test.detach().cpu().numpy())
        test_truelist.append(x_test.detach().cpu().numpy())
        test12_list.append(test12)

        avg_test_loss = (avg_test_loss * idx + test_loss.detach()) / (idx + 1)

    torch.save(test_predlist, os.path.join(results_path, 'reconstructions'))
    torch.save(test_truelist, os.path.join(results_path, 'original'))
    torch.save(test12_list, os.path.join(results_path, 'test12'))

    logger.info("Avg Test Loss = {}".format(avg_test_loss.detach()))
    logger.info("Total Mapping Time = {} seconds".format(mapping_time))

    batches = test_dataset.__len__() / train_params['test_batch']
    min_batch_size = (batches - int(batches)) * train_params['test_batch']

    for i in range(train_params['num_plots']):

        batch = torch.randint(0, len(test_loader), (1, ))

        if batch == len(test_loader) - 1:
            sample = torch.randint(0, int(min_batch_size), (1, ))
        else:
            sample = torch.randint(0, train_params['test_batch'], (1, ))

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)

        for dim in range(128):
            sns.kdeplot(test_truelist[batch][sample][:, dim], ax=ax1)
            sns.kdeplot(test_predlist[batch][sample][:, dim], ax=ax2)

        ax1.set(
            ylabel='Density (unstandardised)',
            title='Original Sample KDE of n=128 Latent Dimensions'
        )
        ax2.set(
            xlabel='Sample Values',
            ylabel='Density (unstandardised)',
            title='Reconstructed Sample KDE of n=128 Latent Dimensions'
        )
        fig.savefig(os.path.join(results_path, f"Test_{i+1}.png"))

        plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        args.model_path, args.results_path, args.training_params, args.matching_params,
        args.train_data_path, args.valid_data_path, args.test_data_path
    )
