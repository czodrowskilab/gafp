# CREAM

CREAM stands for **C**lassification and **Re**gression **a**t **M**erck. 
It's a Python 3 library that makes it easy to train machine learning models 
with Random Forest, XGBoost or neural networks.

## Getting Started
### Prerequisites

The project dependencies are:
* Python >= 3.6
* NumPy >= 1.15
* Scikit-Learn >= 0.20
* RDKit >= 2018.09
* Pandas => 0.23
* TensorFlow => 1.12
* Keras => 2.2
* XGBoost => 0.81

Optionally you need Django 2.1 or higher to use the CREAM REST server.

Of course you also need the code from this repository folder.

### Installing

First of all you need a working Miniconda installation. You can get it at 
https://conda.io/en/latest/miniconda.html.

Now you can create and activate an environment with the following commands:
```bash
conda create -n cream python=3.6
conda activate cream
```

At older Miniconda installations you have to use ``source activate cream`` to activate
the environment.

Note that CREAM itself also works with Python 3.7 but at this point not all 
dependencies are available for a newer Python version than 3.6.

Now you have to install the dependencies with the following command:
```bash
conda install -c defaults -c rdkit -c conda-forge numpy pandas scikit-learn rdkit xgboost tensorflow keras
```

After this you can install CREAM with ``pip install .`` from this folder. 
Check if it works with ``cream --help``.

###Example

Calculate all RDKit desciptors for the example [hERG dataset](data/herg_chembl_fs.sdf):
```bash
cream addprops --sdf herg_chembl_fs.sdf --save-pickle herg.pkl --value-tag pIC50
```

Train a Random Forest classification model with a pIC50 threshold of 5.0
and the prediction labels inactive and active:
```bash
cream categorical --pickle herg.pkl --model-name hERG --thresholds 5.0 --labels inactive,active
```

Predict the training dataset:
```bash
cream predict --model hERG.model --sdf herg_chembl_fs.sdf --save-sdf herg_predictions.sdf
```

## Author

**Marcel Baltruschat** - [GitHub](https://github.com/mrcblt)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

