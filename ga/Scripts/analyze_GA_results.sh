#!/bin/bash
source <path_to_conda_activate_script> <ga_env>

result_dir='results'
mkdir $result_dir

echo "Performances:"


echo "hERG"
cd paper_hERG/splits

echo "TRAIN"
python predictor.py --model=NeuralNet --id=pKi_class --save_csv=train_output.csv paper_hERG/data2model_herg_train_class.sdf paper_hERG/splits/hERG_GA_DNN.config
python Tools/csv_to_kappa.py train_output.csv pKi_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_train_kappa.log

echo "TEST"
python predictor.py --model=NeuralNet --id=pKi_class --save_csv=test_output.csv paper_hERG/data2model_herg_test_class.sdf paper_hERG/splits/hERG_GA_DNN.config
python Tools/csv_to_kappa.py test_output.csv pKi_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_test_kappa.log


echo "CACO"
cd paper_CACO/splits

echo "TRAIN"
python predictor.py --model=NeuralNet --id=papp_class --save_csv=train_output.csv paper_CACO/data2model_caco_train_class.sdf paper_CACO/splits/caco_GA_DNN.config
python Tools/csv_to_kappa.py train_output.csv papp_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/caco_train_kappa.log

echo "TEST"
python predictor.py --model=NeuralNet --id=papp_class --save_csv=test_output.csv paper_CACO/data2model_caco_test_class.sdf paper_CACO/splits/caco_GA_DNN.config
python Tools/csv_to_kappa.py test_output.csv papp_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/caco_test_kappa.log


echo "SOLUBILITY"
cd paper_solubility/splits

echo "TRAIN"
python predictor.py --model=NeuralNet --id=logS_class --save_csv=train_output.csv paper_solubility/data2model_sol_train_class.sdf paper_solubility/splits/sol_GA_DNN.config
python Tools/csv_to_kappa.py train_output.csv logS_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/sol_train_kappa.log

echo "TEST"
python predictor.py --model=NeuralNet --id=logS_class --save_csv=test_output.csv paper_solubility/data2model_sol_test_class.sdf paper_solubility/splits/sol_GA_DNN.config
python Tools/csv_to_kappa.py test_output.csv logS_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/sol_test_kappa.log


echo "CLINT_HUMAN"
cd paper_clint_h/splits

echo "TRAIN"
python predictor.py --model=NeuralNet --id=clint_class --save_csv=train_output.csv paper_clint_h/data2model_clint_hum_train_class.sdf paper_clint_h/splits/clint_h_GA_DNN.config
python Tools/csv_to_kappa.py train_output.csv clint_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/clint_h_train_kappa.log

echo "TEST"
python predictor.py --model=NeuralNet --id=clint_class --save_csv=test_output.csv paper_clint_h/data2model_clint_hum_test_class.sdf paper_clint_h/splits/clint_h_GA_DNN.config
python Tools/csv_to_kappa.py test_output.csv clint_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/clint_h_test_kappa.log


echo "CLINT_MOUSE"
cd paper_clint_m/splits

echo "TRAIN"
python predictor.py --model=NeuralNet --id=clint_class --save_csv=train_output.csv paper_clint_m/data2model_clint_mou_train_class.sdf paper_clint_m/splits/clint_m_GA_DNN.config
python Tools/csv_to_kappa.py train_output.csv clint_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/clint_m_train_kappa.log

echo "TEST"
python predictor.py --model=NeuralNet --id=clint_class --save_csv=test_output.csv paper_clint_m/data2model_clint_mou_test_class.sdf paper_clint_m/splits/clint_m_GA_DNN.config
python Tools/csv_to_kappa.py test_output.csv clint_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/clint_m_test_kappa.log


echo "CLINT_RAT"
cd paper_clint_r/splits

echo "TRAIN"
python predictor.py --model=NeuralNet --id=clint_class --save_csv=train_output.csv paper_clint_r/data2model_clint_rat_train_class.sdf paper_clint_r/splits/clint_r_GA_DNN.config
python Tools/csv_to_kappa.py train_output.csv clint_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/clint_r_train_kappa.log

echo "TEST"
python predictor.py --model=NeuralNet --id=clint_class --save_csv=test_output.csv paper_clint_r/data2model_clint_rat_test_class.sdf paper_clint_r/splits/clint_r_GA_DNN.config
python Tools/csv_to_kappa.py test_output.csv clint_class Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/clint_r_test_kappa.log


echo "hERG_NCV_0"
cd paper_NCV_chembl_herg/outer_loop

echo "TRAIN_0"
python predictor.py --model=NeuralNet --id=TL --save_csv=train_output_0.csv paper_NCV_chembl_herg/outer_loop/trainset_0.sdf paper_NCV_chembl_herg/outer_loop/trainset_0/HERG_CHEMBL_INNER_0.config
python Tools/csv_to_kappa.py train_output_0.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_0_train_kappa.log

echo "TEST_0"
python predictor.py --model=NeuralNet --id=TL --save_csv=test_output_0.csv paper_NCV_chembl_herg/outer_loop/testset_0.sdf paper_NCV_chembl_herg/outer_loop/trainset_0/HERG_CHEMBL_INNER_0.config
python Tools/csv_to_kappa.py test_output_0.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_0_test_kappa.log


echo "hERG_NCV_1"
cd paper_NCV_chembl_herg/outer_loop

echo "TRAIN_1"
python predictor.py --model=NeuralNet --id=TL --save_csv=train_output_1.csv paper_NCV_chembl_herg/outer_loop/trainset_1.sdf paper_NCV_chembl_herg/outer_loop/trainset_1/HERG_CHEMBL_INNER_1.config
python Tools/csv_to_kappa.py train_output_1.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_1_train_kappa.log

echo "TEST_1"
python predictor.py --model=NeuralNet --id=TL --save_csv=test_output_1.csv paper_NCV_chembl_herg/outer_loop/testset_1.sdf paper_NCV_chembl_herg/outer_loop/trainset_1/HERG_CHEMBL_INNER_1.config
python Tools/csv_to_kappa.py test_output_1.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_1_test_kappa.log


echo "hERG_NCV_2"
cd paper_NCV_chembl_herg/outer_loop

echo "TRAIN_2"
python predictor.py --model=NeuralNet --id=TL --save_csv=train_output_2.csv paper_NCV_chembl_herg/outer_loop/trainset_2.sdf paper_NCV_chembl_herg/outer_loop/trainset_2/HERG_CHEMBL_INNER_2.config
python Tools/csv_to_kappa.py train_output_2.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_2_train_kappa.log

echo "TEST_2"
python predictor.py --model=NeuralNet --id=TL --save_csv=test_output_2.csv paper_NCV_chembl_herg/outer_loop/testset_2.sdf paper_NCV_chembl_herg/outer_loop/trainset_2/HERG_CHEMBL_INNER_2.config
python Tools/csv_to_kappa.py test_output_2.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_2_test_kappa.log


echo "hERG_NCV_3"
cd paper_NCV_chembl_herg/outer_loop

echo "TRAIN_3"
python predictor.py --model=NeuralNet --id=TL --save_csv=train_output_3.csv paper_NCV_chembl_herg/outer_loop/trainset_3.sdf paper_NCV_chembl_herg/outer_loop/trainset_3/HERG_CHEMBL_INNER_3.config
python Tools/csv_to_kappa.py train_output_3.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_3_train_kappa.log

echo "TEST_3"
python predictor.py --model=NeuralNet --id=TL --save_csv=test_output_3.csv paper_NCV_chembl_herg/outer_loop/testset_3.sdf paper_NCV_chembl_herg/outer_loop/trainset_3/HERG_CHEMBL_INNER_3.config
python Tools/csv_to_kappa.py test_output_3.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_3_test_kappa.log


echo "hERG_NCV_4"
cd paper_NCV_chembl_herg/outer_loop

echo "TRAIN_4"
python predictor.py --model=NeuralNet --id=TL --save_csv=train_output_4.csv paper_NCV_chembl_herg/outer_loop/trainset_4.sdf paper_NCV_chembl_herg/outer_loop/trainset_4/HERG_CHEMBL_INNER_4.config
python Tools/csv_to_kappa.py train_output_4.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_4_train_kappa.log

echo "TEST_4"
python predictor.py --model=NeuralNet --id=TL --save_csv=test_output_4.csv paper_NCV_chembl_herg/outer_loop/testset_4.sdf paper_NCV_chembl_herg/outer_loop/trainset_4/HERG_CHEMBL_INNER_4.config
python Tools/csv_to_kappa.py test_output_4.csv TL Prediction_0 Prediction_1 Prediction_2 Prediction_3 Prediction_4 > $result_dir/hERG_NCV_4_test_kappa.log
