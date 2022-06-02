# fdsa - Examples

Here we report some `fdsa` usage examples.
Example data can be downloaded [here](https://ibm.box.com/v/paccmann-?).


## Training a Set Matching Network

The set matching network simply plays the role of a mapper in a set-autoencoder and is 
not optimised during training. Therefore, it must first be pre-trained before it can be 
plugged into the set autoencoder.

The synthetic data to train the set matching network can be generated during runtime 
using the `pytoda` package. The dataset can be customised with respect to its 
distribution, shape, and the cost metric used to generate the assignment targets for 
supervised learning. See the training scripts in [set_matching](./set_matching/) folder 
on how to create and use these custom synthetic datasets.

### RNN Model
The example [rnn_setmatching.py](./set_matching/rnn/rnn_setmatching.py) performs a
comparative analysis of RNNs using 4 recurrent cells, namely, GRU, LSTM, nBRC and BRC
in the set matching task. The parameters for training are provided as JSON files.

```console
(fdsa) $ python examples/set_matching/rnn/rnn_setmatching.py -h
usage: rnn_setmatching.py [-h] model_path results_path training_params

positional arguments:
  model_path       Path to save best model.
  results_path     Path to save results and plots.
  training_params  Path to the training parameters.

optional arguments:
  -h, --help       show this help message and exit
```
The RNN models return a set2 vs set1 constrained similarity matrix, and so the predictions
are compared with `target21`.

### Sequence-to-Sequence Model
The example [seq2seq_setmatching.py](./set_matching/seq2seq/seq2seq_setmatching.py) 
performs a comparative analysis of Sequence2Sequence models with attention using 4 
recurrent cells, namely, GRU, LSTM, nBRC and BRC as the encoding/decoding unit in the 
set matching task. The parameters for training are provided as JSON files.

```console
(fdsa) $ python examples/set_matching/seq2seq/seq2seq_setmatching.py -h
usage: seq2seq_setmatching.py [-h] model_path results_path training_params

positional arguments:
  model_path       Path to save best model.
  results_path     Path to save results and plots.
  training_params  Path to the training parameters.

optional arguments:
  -h, --help       show this help message and exit
```

The Seq2Seq models return a set1 vs set2 constrained similarity matrix, and so the 
predictions are compared with `target12`.

NOTE: The dimensions along which the `argmax` is computed in the constrained similarity matrix
decides whether `target12` or `target21` are used as the true targets for loss calculation.
`target12` are the row-wise non-zero indices in a set1 vs set2 constrained similarity matrix,
and `target21` are the row-wise non-zero indices in a set2 vs set1 constrained similarity matrix.



## Training a Fully Differentiable Set Autoencoder.

The [set_autoencoder](./set_autoencoder/) folder contains 2 examples for two tasks -
reconstructing shapes and reconstructing 128D cancer data.

### Reconstructing 2D Shapes
This tasks serves as a sanity check for the set autoencoder and so only uses the Hungarian algorithm.
The data for the shapes task consists of randomly generated 2D point clouds of squares, crosses, and circles,
of various sizes and positions, and made up of a varying number of points. The data can be generated 
using the [shapes_data.py](../fdsa/datasets/shapes_data.py) script. 
The data are saved as a `.csv` file, which is then passed into the [shapes_train.py](./set_autoencoder/shapes/shapes_train.py) 
script. This script performs a comparative analysis of using GRU, LSTM, pLSTM and nBRC recurrent cells
as the encoding/decoding unit in the set autoencoder on reconstructing 2D shapes. The
training parameters are provided as a JSON file.

```console
(fdsa) $ python examples/set_autoencoder/shapes/shapes_train.py -h
usage: shapes_train.py [-h]
                       model_path results_path training_data_path
                       validation_data_path testing_data_path training_params

positional arguments:
  model_path            Path to save the best performing model.
  results_path          Path to save the training and validation losses.
  training_data_path    Path to the training data.
  validation_data_path  Path to the validation data.
  testing_data_path     Path to the testing data.
  training_params       Path to the training parameters json file.

optional arguments:
  -h, --help            show this help message and exit
```
### Reconstructing 128D Cancer Data

The data are provided as `torch` files that contain tensors of transcriptomic and protein data 
combined into sets, and split into training, validation and test datasets, such that no cell line 
or protein seen during training are validated or tested. Additionally, the permutations 
used in shuffling the order of the  elements of the set are also provided. 
The files can be found [here](https://ibm.box.com/v/paccmann-sets-data).

The training script [setsae_setm_train.py](./set_autoencoder/reconstruct_128D/setsae_setm_train.py)
generates reconstructions for the specified parameters, which are provided as JSON files.
The Hungarian or Gale-Shapley algorithm can be used in place of the pre-trained network
as a baseline by commenting and uncommenting the lines pertaining to the mapper.
Note: Even if the experiment is for the Hungarian/Gale-Shapley algorithm, a dummy file for matching parameters and 
model path should be provided to maintain the flexibility of switching between the
network and algorithms.

The pre-trained matching network can be found [here](https://ibm.box.com/v/paccmann-sets-matching-network).

```console
(fdsa) $ python examples/set_autoencoder/reconstruct_128D/setsae_setm_train.py
usage: setsae_setm_train.py [-h]
                            model_path results_path training_params
                            matching_params train_data_path valid_data_path
                            test_data_path

positional arguments:
  model_path       Path where the pre-trained matching network is saved.
  results_path     Path to save the results, logs and best model.
  training_params  Path to the training parameters json file.
  matching_params  Path to the matching network parameters json file.
  train_data_path  Path to the training data.
  valid_data_path  Path to the validation data.
  test_data_path   Path to the testing data.

optional arguments:
  -h, --help       show this help message and exit
```


## Molecule Generation

The example [mol_gen.py](./molecule_generation/mol_gen.py) uses the encoder of the pre-trained set autoencoder to produce embeddings of sets of transcriptomic and proteomic data. This embedding acts as a multi-modal context for molecule generation in the Paccmann^{RL} generative model to generate candidate drugs against a given cancer type. 

The experiment makes use of sets of transcriptomic profiles of cell lines and associated proteins to condition molecule generation. A LOOCV on the cell-lines is used to evaluate the performance of this model, that is, the test set comprises a cartesian product of the test cell-line and all proteins under consideration. 

The [molecule_generation](./molecule_generation) folder contains the necessary JSON parameter files for training. The omics data, protein data and unbiased predictions can be found here [here](https://ibm.box.com/v/paccmann-sets-data). The pre-trained models required to succesfully run the script can be found as follows:
1. Encoder model (encoder_model_path, encoder_params_path) and parameters can be found [here](https://ibm.box.com/v/paccmann-sets-autoencoder)
2. Molecule model (mol_model_path) can be found [here](https://ibm.box.com/v/paccmann-affinity-selfies024)
3. IC50 model (ic50_model_path) can be found [here](https://ibm.box.com/v/paccmann-pytoda-ic50)
4. Affinity model (affinity_model_path) can be found [here](https://ibm.box.com/v/paccmann-affinity-base)
5. Tox21 model (tox21_path) can be found [here](https://ibm.ent.box.com/folder/122603684362?v=paccmann-sarscov2-data)
under pretraining/toxicity_predictor.

```console
(fdsa) $ python examples/molecule_generation/mol_gen.py -h
usage: mol_gen.py [-h] [--test_protein_name TEST_PROTEIN_NAME]
                  [--tox21_path TOX21_PATH]
                  omics_data_path protein_data_path test_cell_line
                  encoder_model_path mol_model_path ic50_model_path
                  affinity_model_path params_path encoder_params_path
                  results_path unbiased_protein_path unbiased_omics_path site
                  model_name

PaccMann^RL training script

positional arguments:
  omics_data_path       Omics data path to condition molecule generation.
  protein_data_path     Protein data path to condition molecule generation.
  test_cell_line        Name of testing cell line (LOOCV).
  encoder_model_path    Path to setAE model.
  mol_model_path        Path to chemistry model.
  ic50_model_path       Path to pretrained IC50 model.
  affinity_model_path   Path to pretrained affinity model.
  params_path           Directory containing the model params JSON file.
  encoder_params_path   directory containing the encoder parameters JSON file.
  results_path          Path where results are saved.
  unbiased_protein_path
                        Path where unbiased protein predictions are saved.
  unbiased_omics_path   Path where unbiased omics predictions are saved.
  site                  Name of the cancer site for conditioning generation.
  model_name            Name for the trained model.

optional arguments:
  -h, --help            show this help message and exit
  --test_protein_name TEST_PROTEIN_NAME
                        Optional gene name of testing protein (LOOCV).
  --tox21_path TOX21_PATH
                        Optional path to Tox21 model.
```

For more examples see other repositories in the [PaccMann organization](https://github.com/PaccMann).

