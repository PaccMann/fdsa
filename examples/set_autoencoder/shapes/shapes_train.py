import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from paccmann_sets.datasets.torch_dataset import Collate, ToySetsDataset
from paccmann_sets.models.sets_autoencoder import SetsAE
from paccmann_sets.utils.hyperparameters import LR_SCHEDULER_FACTORY
from paccmann_sets.utils.loss_setae import SetAELoss
from paccmann_sets.utils.mapper import MapperSetsAE
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    'model_path', type=str, help='Path to save the best performing model.'
)
parser.add_argument(
    'results_path', type=str, help='Path to save the training and validation losses.'
)

parser.add_argument('training_data_path', type=str, help='Path to the training data.')
parser.add_argument(
    'validation_data_path', type=str, help='Path to the validation data.'
)
parser.add_argument('testing_data_path', type=str, help='Path to the testing data.')
parser.add_argument(
    'training_params', type=str, help='Path to the training parameters json file.'
)

# get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(
    model_path: str, results_path: str, training_data_path: str,
    validation_data_path: str, testing_data_path: str, training_params: str
):
    """Main function for reconstructing shapes.

    Args:
        model_path (str): Path to save the best performing model.
        results_path (str): Path to save the training and validation losses.
        training_data_path (str): Path to the training data.
        validation_data_path (str): Path to the validation data.
        testing_data_path (str): Path to the testing data.
        training_params (str): Path to the training parameters json file.
    """

    # setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(results_path, "sets_autoencoder.log")),
            logging.StreamHandler(sys.stdout)
        ],
    )
    logger = logging.getLogger('shapes')
    logger.setLevel(logging.DEBUG)

    clrs = sns.color_palette('Set2', 2)
    fig2, ax2 = plt.subplots(constrained_layout=True)
    ax2.set_title('Train and Validation Loss of Sets AE on Shapes Dataset')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    save_here = os.path.join(model_path, 'shapes_setsae')
    model = None
    if model is not None:
        del (model)

    with open(training_params, 'r') as readjson:
        params = json.load(readjson)

    max_length = params.get('max_length', 33)
    dim = params.get('input_size', 2)
    padding_value = params.get('padding_value', 2.0)

    collator = Collate(max_length, dim, padding_value, device)

    # get training data
    train_dataset = ToySetsDataset(training_data_path, ['x', 'y'], ['label'], ['ID'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['train_batch'],
        shuffle=True,
        collate_fn=collator
    )

    valid_dataset = ToySetsDataset(validation_data_path, ['x', 'y'], ['label'], ['ID'])
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=params['valid_batch'],
        shuffle=True,
        collate_fn=collator
    )

    test_dataset = ToySetsDataset(testing_data_path, ['x', 'y'], ['label'], ['ID'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=params['test_batch'],
        shuffle=True,
        collate_fn=collator
    )

    model = SetsAE(device, **params).to(device)
    mapper = MapperSetsAE(params['matcher'], params['p'], device)
    loss = SetAELoss(params['loss'], device)

    optimiser = torch.optim.Adam(model.parameters(), lr=params['lr'])

    lr_scheduler = LR_SCHEDULER_FACTORY[params['scheduler']
                                        ](optimiser, *params['lr_args'])

    min_loss = np.inf
    epochs = params['epochs']

    plot_train = []
    plot_valid = []
    for epoch in range(epochs):

        model.train()
        logger.info("=== Epoch [{}/{}]".format(epoch + 1, epochs))

        for idx, (x_batch, batch_lengths) in enumerate(train_loader):

            x_batch = x_batch.to(device)

            pred_train, prob_train = model(x_batch, x_batch.size(1), batch_lengths)

            mapped_outputs, mapped_prob, _ = mapper(x_batch, pred_train, prob_train)

            train_loss = loss(x_batch, mapped_outputs, mapped_prob, batch_lengths)

            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()

        cpu_train_loss = train_loss.detach().cpu().numpy()
        plot_train.append(cpu_train_loss)
        logger.info("Train Loss = {}".format(cpu_train_loss))

        model.eval()
        avg_valid_loss = 0
        for idx, (x_valid, batch_lengths) in enumerate(valid_loader):

            x_valid = x_valid.to(device)

            pred_valid, prob_valid = model(x_valid, x_valid.size(1), batch_lengths)

            mapped_outputs_valid, mapped_prob_valid, _ = mapper(
                x_valid, pred_valid, prob_valid
            )

            valid_loss = loss(
                x_valid, mapped_outputs_valid, mapped_prob_valid, batch_lengths
            )

            avg_valid_loss = (avg_valid_loss * idx + valid_loss.detach()) / (idx + 1)

        cpu_avg_valid_loss = avg_valid_loss.detach().cpu().numpy()
        plot_valid.append(cpu_avg_valid_loss)
        logger.info("Avg Valid Loss = {}".format(cpu_avg_valid_loss))

        if avg_valid_loss < min_loss:
            min_loss = avg_valid_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss
                }, save_here
            )
        if params['scheduler'] == 'plateau':
            lr_scheduler.step(valid_loss)
        else:
            lr_scheduler.step()

    np.save(os.path.join(results_path, 'train_loss'), plot_train)
    np.save(os.path.join(results_path, 'avg_valid_loss'), plot_valid)
    ax2.plot(range(len(plot_train)), plot_train, color=clrs[0], label="Train Loss")
    ax2.plot(range(len(plot_valid)), plot_valid, color=clrs[1], label="Valid Loss")
    ax2.legend()
    fig2.savefig(os.path.join(results_path, 'setsAE.png'))

    avg_test_loss = 0
    model.eval()
    test_predlist = []
    test_truelist = []
    for idx, (x_test, batch_lengths) in enumerate(test_loader):

        x_test = x_test.to(device)

        pred_test, prob_test = model(x_test, x_test.size(1), batch_lengths)

        mapped_outputs_test, mapped_prob_test, _ = mapper(x_test, pred_test, prob_test)

        test_loss = loss(x_test, mapped_outputs_test, mapped_prob_test, batch_lengths)

        test_predlist.append(mapped_outputs_test.detach().cpu().numpy())
        test_truelist.append(x_test.detach().cpu().numpy())

        avg_test_loss = (avg_test_loss * idx + test_loss.detach()) / (idx + 1)

    np.save(os.path.join(results_path, 'reconstructions'), test_predlist)
    np.save(os.path.join(results_path, 'original'), test_truelist)
    logger.info("Avg Test Loss = {}".format(avg_test_loss.detach().cpu().numpy()))

    batches = test_dataset.__len__() / params['test_batch']
    min_batch_size = (batches - int(batches)) * params['test_batch']
    for i in range(params['num_plots']):

        batch = torch.randint(0, len(test_loader), (1, ))

        if batch == len(test_loader) - 1:
            sample = torch.randint(0, int(min_batch_size), (1, ))
        else:
            sample = torch.randint(0, params['test_batch'], (1, ))

        fig, ax = plt.subplots()
        ax.set_title("Visualisation of the Original 2D Shape and its Reconstruction")
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.scatter(
            test_truelist[batch][sample][:, 0],
            test_truelist[batch][sample][:, 1],
            label='Original Sample'
        )
        ax.scatter(
            test_predlist[batch][sample][:, 0],
            test_predlist[batch][sample][:, 1],
            label='Reconstructed Sample'
        )
        ax.legend()
        fig.savefig(os.path.join(results_path, f"Test_{i+1}.png"))
        plt.close()


if __name__ == '__main__':
    args = parser.parse_args()

    main(
        args.model_path, args.results_path, args.training_data_path,
        args.validation_data_path, args.testing_data_path, args.training_params
    )
