import argparse
import json
import logging
import os
import sys
import warnings
from itertools import product

import numpy as np
import pandas as pd
import torch

from paccmann_chemistry.models import (StackGRUDecoder, StackGRUEncoder, TeacherVAE)
from paccmann_chemistry.utils import get_device
from paccmann_generator.plot_utils import (
    plot_and_compare, plot_and_compare_proteins, plot_loss
)
from paccmann_generator.reinforce_sets import ReinforceMultiModalSets
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models import MODEL_FACTORY
from fdsa.models.sets_autoencoder import SetsAE
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage, SMILESTokenizer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PaccMann^RL training script')

parser.add_argument(
    'omics_data_path',
    type=str,
    help='Omics data path to condition molecule generation.'
)
parser.add_argument(
    'protein_data_path',
    type=str,
    help='Protein data path to condition molecule generation.'
)

parser.add_argument(
    'test_cell_line', type=str, help='Name of testing cell line (LOOCV).'
)
parser.add_argument('encoder_model_path', type=str, help='Path to setAE model.')

parser.add_argument('mol_model_path', type=str, help='Path to chemistry model.')

parser.add_argument('ic50_model_path', type=str, help='Path to pretrained IC50 model.')

parser.add_argument(
    'affinity_model_path', type=str, help='Path to pretrained affinity model.'
)
parser.add_argument('--tox21_path', help='Optional path to Tox21 model.')

parser.add_argument(
    'params_path', type=str, help='Directory containing the model params JSON file.'
)
parser.add_argument(
    'encoder_params_path',
    type=str,
    help='directory containing the encoder parameters JSON file.'
)
parser.add_argument('results_path', type=str, help='Path where results are saved.')
parser.add_argument(
    'unbiased_protein_path',
    type=str,
    help='Path where unbiased protein predictions are saved.'
)
parser.add_argument(
    'unbiased_omics_path',
    type=str,
    help='Path where unbiased omics predictions are saved.'
)

parser.add_argument(
    'site', type=str, help='Name of the cancer site for conditioning generation.'
)
parser.add_argument('model_name', type=str, help='Name for the trained model.')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('train_paccmann_rl')
logger_m = logging.getLogger('matplotlib')
logger_m.setLevel(logging.WARNING)


def main(*, parser_namespace):

    disable_rdkit_logging()

    # read the params json
    params = dict()
    with open(parser_namespace.params_path) as f:
        params.update(json.load(f))

    with open(parser_namespace.encoder_params_path) as f:
        encoder_params = json.load(f)

    # results_path = params.get('results_path', parser_namespace.results_path)

    mol_model_path = params.get('mol_model_path', parser_namespace.mol_model_path)
    encoder_model_path = params.get(
        'encoder_model_path', parser_namespace.encoder_model_path
    )
    ic50_model_path = params.get('ic50_model_path', parser_namespace.ic50_model_path)
    omics_data_path = params.get('omics_data_path', parser_namespace.omics_data_path)
    affinity_model_path = params.get(
        'affinity_model_path', parser_namespace.affinity_model_path
    )
    protein_data_path = params.get(
        'protein_data_path', parser_namespace.protein_data_path
    )
    model_name = params.get(
        'model_name', parser_namespace.model_name
    )   # yapf: disable

    unbiased_protein_path = params.get(
        'unbiased_protein_path', parser_namespace.unbiased_protein_path
    )   # yapf: disable
    unbiased_omics_path = params.get(
        'unbiased_omics_path', parser_namespace.unbiased_omics_path
    )   # yapf: disable
    site = params.get(
        'site', parser_namespace.site
    )   # yapf: disable

    test_cell_line = params.get('test_cell_line', parser_namespace.test_cell_line)

    logger.info(f'Model with name {model_name} starts.')

    # passing optional paths to params to possibly update_reward_fn
    optional_reward_args = ['tox21_path', 'site']
    for arg in optional_reward_args:
        if parser_namespace.__dict__[arg]:
            params[arg] = params.get(arg, parser_namespace.__dict__[arg])

    omics_df = pd.read_csv(omics_data_path)

    protein_df = pd.read_csv(protein_data_path)
    protein_df.index = protein_df['entry_name']

    # Restore SMILES Model
    with open(os.path.join(mol_model_path, 'model_params.json')) as f:
        mol_params = json.load(f)

    gru_encoder = StackGRUEncoder(mol_params)
    gru_decoder = StackGRUDecoder(mol_params)
    generator = TeacherVAE(gru_encoder, gru_decoder)
    generator.load(
        os.path.join(
            mol_model_path, f"weights/best_{params.get('smiles_metric', 'rec')}.pt"
        ),
        map_location=get_device()
    )

    # Load languages

    generator_smiles_language = SMILESTokenizer(
        vocab_file=os.path.join(mol_model_path, 'vocab.json')
    )

    generator.smiles_language = generator_smiles_language

    #load predictors
    with open(os.path.join(ic50_model_path, 'model_params.json')) as f:
        paccmann_params = json.load(f)

    paccmann_predictor = MODEL_FACTORY['mca'](paccmann_params)
    paccmann_predictor.load(
        os.path.join(
            ic50_model_path, f"weights/best_{params.get('ic50_metric', 'rmse')}_mca.pt"
        ),
        map_location=get_device()
    )
    paccmann_predictor.eval()

    paccmann_smiles_language = SMILESLanguage.from_pretrained(
        pretrained_path=ic50_model_path
    )

    paccmann_predictor._associate_language(paccmann_smiles_language)

    with open(os.path.join(affinity_model_path, 'model_params.json')) as f:
        protein_pred_params = json.load(f)

    protein_predictor = MODEL_FACTORY['bimodal_mca'](protein_pred_params)
    protein_predictor.load(
        os.path.join(
            affinity_model_path,
            f"weights/best_{params.get('p_metric', 'ROC-AUC')}_bimodal_mca.pt"
        ),
        map_location=get_device()
    )
    protein_predictor.eval()

    affinity_smiles_language = SMILESLanguage.from_pretrained(
        pretrained_path=os.path.join(affinity_model_path, 'smiles_serial')
    )
    affinity_protein_language = ProteinLanguage()

    protein_predictor._associate_language(affinity_smiles_language)
    protein_predictor._associate_language(affinity_protein_language)

    setsae = SetsAE(device, **encoder_params).to(device)

    setsae.load_state_dict(torch.load(encoder_model_path, map_location=get_device()))

    set_encoder = setsae.encoder
    set_encoder.latent_size = set_encoder.hidden_size_encoder

    #############################################
    # Create a generator model that will be optimized
    gru_encoder_rl = StackGRUEncoder(mol_params)
    gru_decoder_rl = StackGRUDecoder(mol_params)
    generator_rl = TeacherVAE(gru_encoder_rl, gru_decoder_rl)
    generator_rl.load(
        os.path.join(mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"),
        map_location=get_device()
    )
    generator_rl.smiles_language = generator_smiles_language
    generator_rl.eval()
    # generator

    model_folder_name = test_cell_line + '_' + 'SetAE'

    learner = ReinforceMultiModalSets(
        generator_rl, set_encoder, protein_predictor, paccmann_predictor, protein_df,
        omics_df, params, generator_smiles_language, model_folder_name, logger, True
    )

    train_omics = omics_df[omics_df['cell_line'] != test_cell_line]['cell_line']
    train_protein = protein_df['entry_name']

    # train_sets = list(product(train_omics, train_protein))
    test_sets = list(product([test_cell_line], train_protein))
    assert len(test_sets) == len(protein_df)

    unbiased_preds_ic50 = np.array(
        pd.read_csv(os.path.join(unbiased_omics_path,
                                 test_cell_line + '.csv'))['IC50'].values
    )

    biased_efficacy_ratios, biased_affinity_ratios, tox_ratios = [], [], []
    rewards, rl_losses = [], []
    gen_mols, gen_prot, gen_cell = [], [], []
    gen_affinity, gen_ic50, modes = [], [], []
    proteins_tested = []
    batch_size = params['batch_size']

    logger.info(f'Model stored at {learner.model_path}')
    # total_train = len(train_sets)

    protein_name = None
    for epoch in range(1, params['epochs'] + 1):
        logger.info(f"Epoch {epoch:d}/{params['epochs']:d}")

        for step in range(1, params['steps'] + 1):
            cell_line = np.random.choice(train_omics)
            protein_name = np.random.choice(train_protein)
            # sample = np.random.randint(total_train)
            # cell_line, protein_name = train_sets[sample]

            logger.info(f'Current train cell: {cell_line}')
            logger.info(f'Current train protein: {protein_name}')

            rew, loss = learner.policy_gradient(
                cell_line, protein_name, epoch, batch_size
            )
            logger.info(
                f"Step {step:d}/{params['steps']:d} \t loss={loss:.2f}, mean rew={rew:.2f}"
            )

            rewards.append(rew.item())
            rl_losses.append(loss)

        # Save model
        if epoch % 5 == 0:
            learner.save(f'gen_{epoch}.pt', f'enc_{epoch}.pt')

        # unbiased pred files are given by protein accession number, so convert entry_name
        protein_accession = protein_df.loc[protein_name, 'accession_number']
        train_unbiased_preds_affinity = np.array(
            pd.read_csv(
                os.path.join(unbiased_protein_path, protein_accession + '.csv')
            )['affinity'].values
        )

        train_unbiased_preds_ic50 = np.array(
            pd.read_csv(os.path.join(unbiased_omics_path,
                                     cell_line + '.csv'))['IC50'].values
        )

        smiles, preds_affinity, preds_ic50, idx = learner.generate_compounds_and_evaluate(
            epoch, params['eval_batch_size'], protein_name, cell_line
        )

        gs = [
            s for i, s in enumerate(smiles)
            if preds_ic50[i] < learner.ic50_threshold and preds_affinity[i] > 0.5
        ]

        gp_ic50 = preds_ic50[(preds_ic50 < learner.ic50_threshold)
                             & (preds_affinity > 0.5)]
        gp_affinity = preds_affinity[(preds_ic50 < learner.ic50_threshold)
                                     & (preds_affinity > 0.5)]

        for ic50, affinity, s in zip(gp_ic50, gp_affinity, gs):
            gen_mols.append(s)
            gen_cell.append(cell_line)
            gen_prot.append(protein_name)
            gen_affinity.append(affinity)
            gen_ic50.append(ic50)
            modes.append('train')

        plot_and_compare_proteins(
            train_unbiased_preds_affinity, preds_affinity, protein_name, epoch,
            learner.model_path, 'train', params['eval_batch_size']
        )

        plot_and_compare(
            train_unbiased_preds_ic50, preds_ic50, site, cell_line, epoch,
            learner.model_path, 'train', params['eval_batch_size']
        )

        # test_cell_line = np.random.choice(test_omics)
        # test_protein_name = np.random.choice(test_protein)
        if epoch > 10 and epoch % 5 == 0:
            for test_idx, test_sample in enumerate(test_sets):
                test_cell_line, test_protein_name = test_sample
                proteins_tested.append(test_protein_name)
                logger.info(f'EVAL cell: {test_cell_line}')
                logger.info(f'EVAL protein: {test_protein_name}')

                test_protein_accession = protein_df.loc[test_protein_name,
                                                        'accession_number']
                unbiased_preds_affinity = np.array(
                    pd.read_csv(
                        os.path.join(
                            unbiased_protein_path, test_protein_accession + '.csv'
                        )
                    )['affinity'].values
                )

                smiles, preds_affinity, preds_ic50, idx = (
                    learner.generate_compounds_and_evaluate(
                        epoch, params['eval_batch_size'], test_protein_name,
                        test_cell_line
                    )
                )

                gs = [
                    s for i, s in enumerate(smiles) if
                    preds_ic50[i] < learner.ic50_threshold and preds_affinity[i] > 0.5
                ]

                gp_ic50 = preds_ic50[(preds_ic50 < learner.ic50_threshold)
                                     & (preds_affinity > 0.5)]
                gp_affinity = preds_affinity[(preds_ic50 < learner.ic50_threshold)
                                             & (preds_affinity > 0.5)]

                for ic50, affinity, s in zip(gp_ic50, gp_affinity, gs):
                    gen_mols.append(s)
                    gen_cell.append(test_cell_line)
                    gen_prot.append(test_protein_name)
                    gen_affinity.append(affinity)
                    gen_ic50.append(ic50)
                    modes.append('test')

                inds = np.argsort(gp_ic50)[::-1]
                for i in inds[:5]:
                    logger.info(
                        f'Epoch {epoch:d}, generated {gs[i]} against '
                        f'{test_protein_name} and {test_cell_line}.\n'
                        f'Predicted IC50 = {gp_ic50[i]}, Predicted Affinity = {gp_affinity[i]}.'
                    )

                plot_and_compare(
                    unbiased_preds_ic50, preds_ic50, site, test_cell_line, epoch,
                    learner.model_path, f'test_{test_protein_name}',
                    params['eval_batch_size']
                )

                plot_and_compare_proteins(
                    unbiased_preds_affinity, preds_affinity, test_protein_name, epoch,
                    learner.model_path, 'test', params['eval_batch_size']
                )

                biased_affinity_ratios.append(
                    np.round(
                        100 * (np.sum(preds_affinity > 0.5) / len(preds_affinity)), 1
                    )
                )

                biased_efficacy_ratios.append(
                    np.round(
                        100 *
                        (np.sum(preds_ic50 < learner.ic50_threshold) / len(preds_ic50)),
                        1
                    )
                )

                all_toxes = np.array([learner.tox21(s) for s in smiles])
                tox_ratios.append(
                    np.round(100 * (np.sum(all_toxes == 1.) / len(all_toxes)), 1)
                )
                logger.info(f'Percentage of non-toxic compounds {tox_ratios[-1]}')

                toxes = [learner.tox21(s) for s in gen_mols]
                # Save results (good molecules!) in DF
                df = pd.DataFrame(
                    {
                        'protein': gen_prot,
                        'cell_line': gen_cell,
                        'SMILES': gen_mols,
                        'IC50': gen_ic50,
                        'Binding probability': gen_affinity,
                        'mode': modes,
                        'Tox21': toxes
                    }
                )

                df.to_csv(os.path.join(learner.model_path, 'results', 'generated.csv'))
                # Plot loss development
                loss_df = pd.DataFrame({'loss': rl_losses, 'rewards': rewards})
                loss_df.to_csv(
                    learner.model_path + '/results/loss_reward_evolution.csv'
                )

    pd.DataFrame(
        {
            'proteins': proteins_tested,
            'efficacy_ratio': biased_efficacy_ratios,
            'affinity_ratio': biased_affinity_ratios,
            'tox_ratio': tox_ratios
        }
    ).to_csv(learner.model_path + '/results/ratios.csv')

    rewards_p_all = loss_df['rewards']
    losses_p_all = loss_df['loss']
    plot_loss(
        losses_p_all, rewards_p_all, params['epochs'], learner.model_path, rolling=5
    )


if __name__ == '__main__':
    main(parser_namespace=args)
