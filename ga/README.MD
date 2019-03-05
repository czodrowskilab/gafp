# Genetic Searcher

This is a genetic algorithm to optimize neural network architectures for specific datasets to find good fitting predictors.

## Getting Started
### Prerequisites

* Python: 3.5.5
* Numpy: 1.13.3
* Scikit-Learn: 0.18.1
* RDKit: 2018.03.2
* Pandas: 0.23.0
* Keras: 2.0.2
* Theano: 0.9
* CUDA: 7.5
* CUDNN: 5.1.03

More recent versions of the libraries may work but are not tested.

### Installing

First of all you need a working Miniconda installation. You can get it at
https://conda.io/en/latest/miniconda.html.

To create the conda environment and install all requirements please use:

```bash
conda create --name ga_env --file install_env.txt
```

Activate the environment:

```bash
source activate ga_env
```

### Dataset preparation

#### Convert floating point based activity values to binary
We provide a small tool to classify molecules.

Usage: ```python classify.py <file.sdf> <property_label> <cutoffs> ```

This generates a new file ```<file>_class.sdf ``` with a property ```<property>_class```.

The syntax to describe the cutoffs is
```<class_id>:<lambda expression>;<class_id>:<lambda expression>```. The value within the lambda is a float represented by x and the complete expression should return a boolean.

Example:
```bash
python classify.py dataset.sdf pKi '0:x<5.0;1:x>=5.0'
```
The newly created file ```dataset_class.sdf``` contains molecules with a new label ```pKi_class```.

#### Split dataset into train-/ testfolds
We provide a tool to split a SD-File into multiple train-/ testfolds, using scikit-learn StratifiedKFold.

Usage:
```bash
python sdf_splitter.py [-h] [--folder FOLDER] [--splitfold SPLITFOLD]
                       SDF label_col

Splits SDF with annotated classes into n files with same class ratio.

positional arguments:
  SDF                   Sdf to use for split
  label_col             Property with class-label

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER       Folder to save output-files
  --splitfold SPLITFOLD
                        how many splits should be done
 ```


### Using the Genetic Searcher
Usage:
``` bash
python gs.py [-h] [--sdf SDF] [--model MODEL] [--label_col LABEL_COL]
             [--folder FOLDER] [--model_folder MODEL_FOLDER]
             [--splitfold SPLITFOLD] [--metric METRIC] [--workers WORKERS]
             [--population POPULATION] [--generations GENERATIONS] [--silent]
             [--log LOG] [--fp_size FP_SIZE]
             [--smarts_patterns SMARTS_PATTERNS] [--descriptors DESCRIPTORS]
             [--wrapper WRAPPER] [--external_test EXTERNAL_TEST] [--local]
             [--local_gpu_dev_ids LOCAL_GPU_DEV_IDS]
             [--mutation_rate_setting {0,1}] [--name NAME]

optional arguments:
  -h, --help            Show this help message and exit
  --sdf SDF             SD-File to use (default: None)
  --model MODEL         Model type to use: NeuralNet or GradientBoost
                        (default: NeuralNet)
  --label_col LABEL_COL
                        Property with classes (default: None)
  --folder FOLDER       Folder to load/save new sdf-files and pickle-files.
                        Filenames of train- and test-files have to start with "train" or "test".
                        (default: None)
  --model_folder MODEL_FOLDER
                        Folder to save model-files and configuration-file.
                        (default: .)
  --splitfold SPLITFOLD
                        How many splits should be done (default: 5)
  --metric METRIC       Metric function to evaluate performance of a model
                        (default: kappa)
  --workers WORKERS     Number of cluster-workers (default: 10)
  --population POPULATION
                        Number of entities per generation of the genetic alg
                        (default: 100)
  --generations GENERATIONS
                        Number of generations for the genetic alg, minimum of
                        3 (default: 100)
  --log LOG             Name of the csv-log file. (default: logfile_%Y_%m_%d__%H_%M_%S.csv)
  --fp_size FP_SIZE     Size of the fingerprint. (default: 4096)
  --smarts_patterns SMARTS_PATTERNS
                        File with smarts-patterns. (default: NO_SMARTS)
  --descriptors DESCRIPTORS
                        Use 'all' or a comma-separated list of descriptors.
                        (default: NO_DESCRIPTORS)
  --wrapper WRAPPER     Argument to indicate wrapper, so that desc calc is
                        bypassed (default: False)
  --external_test EXTERNAL_TEST
                        Argument to run an additional test during the predictor-evaluation
                        using this given dataset. (default: None)
  --local               Try to run locally (on something with a gpu) (default: False)
  --local_gpu_dev_ids LOCAL_GPU_DEV_IDS
                        Comma-separated list of gpu-ids to use (use together with
                        --local) (default: None)
  --mutation_rate_setting {0,1}
                        0: default mutation_rate (crossing_over_rate=0.3,
                        layer_add_rate=0.1, layer_drop_rate=0.1,
                        mutation_rate=0.05, mutation_strength=1) 1: increased
                        mutation_rate (crossing_over_rate=0.3,
                        layer_add_rate=0.1, layer_drop_rate=0.1,
                        mutation_rate=0.1, mutation_strength=2) (default: 0)
  --name NAME           Name of the Model. (default: None)
```

You should run the GA on a head-node of a GPU-cluster or on a server with GPU-Cards. We provide examples for both situations below.
We recommend to split a SD-File into k folds using our script sdf_splitter.py and using the option --folder of the Genetic Searcher to provide a folder containing the SD-Files.

### Predict molecules using the resulting predictors
Usage: |
```bash
predictor.py [-h] [--model MODEL] [--id ID] [--pred_col PRED_COL]
                    [--wrapper WRAPPER] [--save_csv SAVE_CSV]
                    [--save_sdf SAVE_SDF] [--write_all]
                    sdf model_config [model_config ...]

Predicts all molecules using a NeuronalNetwork.

positional arguments:
  sdf                  SD-File to predict.
  model_config         Path to model_config file (result of a GA-run).

optional arguments:
  -h, --help           show this help message and exit
  --model MODEL        Model type to use: NeuralNet or GradientBoost.
                       (default: NeuralNet)
  --id ID              Id-column of the SD-File. (default: None)
  --pred_col PRED_COL  Prefix of new columns. (default: Prediction)
  --wrapper WRAPPER    Wrapper argument to override the descriptor
                       calculation. (default: False)
  --save_csv SAVE_CSV  Write results into this csv_file. (default: output.csv)
  --save_sdf SAVE_SDF  Write results into this SD-File. (default: None)
  --write_all          Write all existing properties of the input SD-File into
                       the resulting file. (default: False)
```

### Example
1. Download a dataset (_e.g._ [hERG](https://github.com/czodrowskilab/gafp/blob/master/fp/data/herg_chembl_fs.sdf))
    ```bash
    wget https://github.com/czodrowskilab/gafp/blob/master/fp/data/herg_chembl_fs.sdf
    ```

2. Convert string-based labels (inactive/active) labels to digits (0/1):
    ```bash
    sed 's/inactive/0/g' herg_chembl_fs.sdf | sed 's/active/1/g' > herg_chembl_fs_digit.sdf
    ```

3. Split dataset into 5 folds:|
    ```bash
    python Tools/sdf_splitter.py --folder=datasets --splitfold=5 herg_chembl_fs_digit.sdf hERG_class
    ```

4. Run the Genetic Searcher to find good performing NN architectures (it also saves the best NNs):
    ```bash
    cd datasets
    python ../gs.py --folder=. --fp_size=4096 --workers=4 --local --local_gpu_dev_ids=0,1,2,3 --population=100 --generations=100 --label_col=hERG_class --name=herg_testrun
    ```

5. Apply the best found model to get predictions:
    ```bash
    python predictor.py --model=NeuralNet --id=hERG_class --save_csv=pred.csv other_ds/testset_1.sdf herg_testrun.config
    ```

6. Inspect the results:
    ```bash
   >head pred.csv
    hERG_class;SD;UniSMILES;pIC50;Prediction_0
    1;0.0;CCN1C[C@H]2CCCC[C@]2(C1)c1ccc(Cl)c(Cl)c1;5.33;1
    1;0.0;CCN1C[C@H]2C[C@]2(C1)c1ccc(NS(=O)(=O)c2ccc(cc2)C(C)C)cc1;5.9;1
    1;0.0;CCN1C[C@H]2[C@H](CN(Cc3cccc(OC(F)(F)F)c3)C(=O)c3cn(C)cn3)[C@H]2C1;5.66;1
    1;0.0;CCNC(=O)N1CC2(CCN(CC3CCN(CC3)C(=O)OCC)CC2)c2cccnc12;6.64;1
    1;0.0;CCNC(=O)N1CCN(CC1)c1cnc2cc(cc(NCc3cccc(c3)[N+]([O-])=O)c2c1)C(F)(F)F;6.68;1
    1;0.0;CCNC(=O)Nc1ccc(cc1)-c1nc2CN(CCc2c(n1)N1CCOC[C@@H]1C)C(C)C;5.07;1
    1;0.0;CCNC(=O)c1ccc(cc1)-c1ccc2C(=O)N(CCN3CCC[C@H]3C)CCc2c1;5.13;1
    1;0.0;CCNCc1cc(Nc2ccnc3cc(Cl)ccc23)ccc1O;5.02;1
    1;0.0;CCNCc1cc(cc(c1)C(F)(F)F)C(=O)N[C@@H]1CCc2ccc(Oc3ccnc4NC(=O)CCc34)cc2C1;5.25;1
    ```
## Author

**Wolf-Guido Bolick** - [GitHub](https://github.com/maltos)
