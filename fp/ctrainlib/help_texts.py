PARSER_DESC = '''
Use RDKit descriptors and scikit-learn learning algorithms
to train categorical and continuous models and make predictions.

                  Subcommand 'listprops'

  1) Get a reminder of what descriptors RDKit can compute.

  See 'cream listprops --help' for details.

                  Subcommand 'listfps'

  2) Get a reminder of the predefined fingerprint aliases and their
  full fingerprint type definition. (The 'addprops' -fp option also 
  lets you specify your own definitions.)

  See 'cream listfps --help' for details.

                  Subcommand 'addprops'

  3) Start with a SD file, where the data to predict is stored in one
  of the tags.

  4) Process the file and use RDKit to compute descriptors and 
  fingerprints used for training.

  5) Combine the structure data, the data from the SD tag, and
  the RDKit descriptors into a Pandas table, which will be used to
  make model.

  See 'cream addprops --help' for details.

                  Subcommand 'categorical'

  6a) Specify the classifier and parameters to train a classification model.

  See 'cream categorical --help' for details.
  
  6b) Specify the regressor and parameters to train a continuous model.
  
  See 'cream continuous --help' for details.

                  Subcommand 'predict'

  7) To make a prediction, start with a SD file and specify a model
  path.

  8) Use RDKit to compute the needed descriptors and fingerprints.

  9) Use the model to make a prediction and get the probabilities.

  10) Save the results to a SD file, using new tags for
  the predicted class and probabilities, or the predicted value in case
  of a continuous model.
  
  See 'cream predict --help' for details.
  
                  Subcommand 'makefpcdb'
                  
  11) Create a fingerprint and compound database from a SD file.
  
  See 'cream makefpcdb --help' for details.

'''

PARSER_EPI = '''
Examples:

To create a Pandas table with the structures and 'pIC50' tag from a
SD file, and with all of the computed RDKit descriptors:

   cream addprops --sdf hERG_pIC50.sdf --value-tag pIC50 
      --save-pickle hERG_pIC50.props.pkl

For details on how to generate fingerprints, use 'cream addprops --help'
and look at the description of the '-fp' option.

To train a categorical model using the RandomForest classifier against
the 'pIC50' column from the Pandas data table, where values <= 5.0 are
'inactive' and values > 5.0 are 'active':

   cream categorical --model-name hERG_class --pickle hERG_pIC50.props.pkl
      --thresholds 5 --labels inactive,active
      
To train a continuous model using the RandomForest regressor against the
'pIC50' column from the Pandas data table:

   cream continuous --model-name hERG_pIC50 --pickle hERG_pIC50.props.pkl

To make a prediction using a trained model, given a SD file (the
required RDKit properties will be computed automatically):

   cream predict --sdf important_project.sdf --model-file hERG_pIC50.model
     --save-sdf important_project.pred.sdf

'''

LISTPROPS_DESC = '''
Use this subcommand to get information about the available RDKit
descriptor names.

'''

LISTPROPS_EPI = '''
The output is a list with all available RDKit descriptors.

'''

LISTFPS_DESC = '''
Use this subcommand to get information about the available fingerprints
and their corresponding parameters.

'''

LISTFPS_EPI = '''
The output is a list with all available fingerprint algorithms and their
corresponding default parameters. These parameters can be adjusted during
"addprops".

'''

ADDPROPS_DESC = '''
Create a pickle file which contains the value tag data from a
SD file with computed RDKit descriptors and fingerprints for each 
structure and the chosen fingerprint and descriptor definitions.

'''

ADDPROPS_EPI = '''
The options can be grouped as tag processing,
descriptor processing, and I/O processing.

  -- Processing the tag with prediction values --

Use --value-tag to specify which tag contains the value to predict.
Its value will be treated as a float.

  -- RDKit descriptor processing --

If you do not specify a descriptor or a fingerprint then
'addprops' will compute all of the RDKit descriptors.

If you have a fingerprint then you must use '--all-descriptors'
to have cream compute all of the descriptors.

Use '--descriptor' (or the shorter '-d' alias) to have cream compute
the named RDKit descriptor for each molecule. This must be specified
once for each descriptor.

  -- RDKit fingerprint processing --

Use the '-fp' option to include boolean fingerprint columns
corresponding to the bit values in a fingerprint. Each fingerprint has
an alias and a fingerprint type. One of the built-in aliases is
"MACCS166", for the MACCS 166 key fingerprint. Use "-fp MACCS166" to
add the columns "MACCS166[0]", "MACCS166[1]", ... "MACCS166[165]",
which correspond to the values for key 1, key 2, .. key 166,
respectively. The '-fp' can be specified multiple times to generate a
fusion fingerprint.

Use the 'listfps' command to get the list of available
fingerprints. If you want to adjust the default parameters you can also 
use the '-fp' command to define your own fingerprint configuration 
with the syntax ALIAS="FP PARAM1=X PARAM2=Y".

For example, the following creates the new alias 'short' defined as
the 128 bit RDKit hash fingerprint.

   cream addprops --sdf a.sdf -fp short="RDKit fpSize=128" 
     --save-pickle a.pkl

It will add the 128 columns "short[0]", "short[1]", ... "short[127]"
to the table. 

  -- I/O processing --

Use --sdf to specify the input SD file. Use --save-pickle to save the 
dataframe and all needed information for training and later prediction 
as a pickle file.

Examples:

To create a Pandas table with the structures and 'pIC50' tag from a
SD file, and with all of the computed RDKit descriptors:

   cream addprops --sdf hERG_pIC50.sdf --value-tag pIC50 
      --save-pickle hERG_pIC50.props.pkl

'''

CATEGORICAL_DESC = '''
Use the pickle file from "cream addprops" and train a categorical
model with the chosen parameters.

'''

CATEGORICAL_EPI = '''
This subcommand trains a categorical model. It first checks all
given parameters, then loads the pickle file generated by "addprops"
and finally starts the training. This includes also clustering,
fingerprint filtering or feature scaling, depending of the chosen
parameters.

The model for --model-name ABC is stored in the pickle file "ABC.model".

The during "addprops" specified value tag must be converted into categories 
for training. The --thresholds options specifies the internal edges, so 
"--thresholds 24.5" describes two categories where values < 24.5 are in 
category "0" and >= 24.5 are in category "1". Use something like 
"--labels inactive,active" if you want more descriptive labels.

If you want to use negative thresholds you have to specify threshold values
separately. Take a look at the examples at the end of this help. The
specified thresholds are sorted from low to high.

Use --classifier to choose something other than RandomForest. Each classifier 
has its own set of parameters, which you can specify with --option. For example, 
to use the XGradientBoost with the 'dart' booster, and using the short-hand
option '-o', you can specify:

      -c XGradientBoost -o booster=dart

If you want to train a model that uses k-models from k-fold cross validation
to perform predictions (CVClassifier) use --cv-classifier followed by number 
of folds (K_FOLDS) you want to use. It uses classifier specified with --classifier
and its' options (--option) as base classifier. The minimum number of folds are 3.

If you want to train a model that uses nested cluster cross validation use
--cluster-classifier followed by a number of clusters. The minimum number of clusters
are 3. To perform this the descriptors and fingerprints are used to cluster the data
into the specified number of clusters. Then these clusters are used to make a cross
validation with the same number of folds as the number of clusters. For every fold
the remaining clusters are used to build the inner folds. For example if you have five
clusters, altogether 5 * 4 models are trained and used for predictions.

If you want to scale the descriptor data (not the fingerprint bits) with z-scaling
you can use --feature-scaling.

If you want to filter the fingerprint bits by variance you can use --fingerprint-filter
followed by a float value between 0.0 and 1.0 to specify the minimum needed variance
within one fingerprint bit column to keep it for training, or instead of a float value
you use "auto" to find the best variance threshold automatically. This is done by
testing a range of thresholds between 0.0 and 0.1 with five fold cross validation.

Example:

To train a categorical model using the default RandomForest classifier
against the 'pIC50' column from the Pandas data file, where values <
5.0 are "inactive" and values >= 5.0 are "active":

   cream categorical --model-name herg --pickle hERG_pIC50.props.pkl
      --thresholds 5 --labels inactive,active

If you want to use CVClassifier using the same classifier as above as base classifier
with 5-fold cross validation:

   cream categorical --model-name herg --pickle hERG_pIC50.props.pkl
      --thresholds 5 --labels inactive,active --cv-classifier 5

If you want to use negative thresholds:

    cream categorical --model-name clint_human --pickle clint_props.pkl
      --thresholds -30 --thresholds -100

If you want to use multiple, not negative thresholds you can specify them by comma:

    cream categorical --model-name clint_human --pickle clint_props.pkl
      --thresholds 30,100
      
If you want to use feature scaling, fingerprint filtering with a variance threshold of
0.01 and a nested cluster cross validation, you could use the following command:

    cream categorical --model-name clint --pickle clint_props.pkl --thresholds 30,100
      --cluster-classifier 5 --feature-scaling --fingerprint-filter 0.01

'''

PREDICT_DESC = '''
Read structures from a SD file, compute all needed descriptors and fingerprints, make
a prediction based on a model, and save the result as new tags to a new SD file.

'''

PREDICT_EPI = '''
The input structures come from the --sdf file and --model specifies 
the model file to use.

The prediction category, or value in case of a continuous model,
is saved in the tag '$MODEL_prediction'. If the model is a classifier,
then its probability is saved in "$MODEL_probability" and the individual 
class prediction probabilities are in $MODEL_probability_0,
$MODEL_probability_1, etc. 

Example:

To make a prediction using a trained model, given a SD file (the
required RDKit properties will be computed automatically):

   cream predict --sdf important_project.sdf --model herg.model
     --save-sdf important_project.pred.sdf

'''

MAKEFPCDB_DESC = '''
Read structures from a SD file, compute the specified fingerprint (default: "RDKit")
for all structures and save the fingerprints, SMILES, molecule IDs and (optional)
float values from a given value tag to a SQLite database. The fingerprint definition
is also saved to the database.

'''

MAKEFPCDB_EPI = '''
The input structures come from the --sdf file and --fingerprint specifies the
fingerprint which has to be computed for all structures from the input file.

The database contains two tables, "fpcdb" and "settings".
The "fpcdb" column names and data types:

| id     | smiles | fp     | value  |
| ------ | ------ | ------ | ------ |
| <TEXT> | <TEXT> | <BLOB> | <REAL> |

The "settings" column names and data types:

| name   | value  |
| ------ | ------ |
| <TEXT> | <TEXT> |

The fingerprint can be defined like in "addprops". For example, the following 
creates the new alias 'short' defined as the 128 bit RDKit hash fingerprint.

   cream makefpcdb --sdf a.sdf -fp short="RDKit fpSize=128" 
     --save-fpcdb example.db --value-tag pIC50
     
The fingerprint definition will be saved in the settings table as pickled
FingerprintInfo instance (name: "fingerprint_info").

'''

CONTINUOUS_DESC = '''
Use the pickle file from "cream addprops" and train a continuous
model with the chosen parameters.

'''

CONTINUOUS_EPI = '''
This subcommand trains a continuous model. It first checks all
given parameters, then loads the pickle file generated by "addprops"
and finally starts the training. This includes fingerprint filtering or 
feature scaling, depending of the chosen parameters.

The model for --model-name ABC is stored in the pickle file "ABC.model".

Use --regressor to choose something other than RandomForest. Each regressor 
has its own set of parameters, which you can specify with --option. For example, 
to use the RandomForest with 1000 trees, and using the short-hand
option '-o', you can specify:

      -o n_estimators=1000

If you want to scale the descriptor data (not the fingerprint bits) with z-scaling
you can use --feature-scaling.

If you want to filter the fingerprint bits by variance you can use --fingerprint-filter
followed by a float value between 0.0 and 1.0 to specify the minimum needed variance
within one fingerprint bit column to keep it for training, or instead of a float value
you use "auto" to find the best variance threshold automatically. This is done by
testing a range of thresholds between 0.0 and 0.1 with five fold cross validation.

Example:

To train a continuous model using the default RandomForest regressor
against the 'pIC50' column from the Pandas data file:

   cream continuous --model-name herg --pickle hERG_pIC50.props.pkl
      
If you want to use feature scaling, fingerprint filtering with a variance threshold of
0.01, you could use the following command:

    cream continuous --model-name clint --pickle clint_props.pkl 
      --feature-scaling --fingerprint-filter 0.01

'''