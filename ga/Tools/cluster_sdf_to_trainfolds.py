import os,sys
import rdkit
from rdkit import Chem
import pathlib
import itertools


def split_to_folds(filename,label):
    class_molecules = dict()
    class_molecules_keyset = set()
    class_molecules_keyset.add(0)
    class_molecules_keyset.add(1)
    class_molecules_keyset.add(2)
    class_molecules_keyset.add(3)
    class_molecules_keyset.add(4)



    # sdm = Chem.SDMolSupplier(filename, sanitize = False, removeHs= False, strictParsing=False)
    # for mol in sdm:
    #     mol_label = mol.GetProp(label)
    #     if not mol_label in class_molecules_keyset:
    #         class_molecules_keyset.add(mol_label)
    #         class_molecules[mol_label] = []
    #     class_molecules[mol_label].append(mol)
    # pass
    # for k in class_molecules_keyset:
    #     print("cluster {} has {} mols".format(k,len(class_molecules[k])))

    train_filename_temp = "trainset_{}.sdf"
    test_filename_temp = "testset_{}.sdf"
    inner_folder = "trainset_{}"
    for k in class_molecules_keyset: # outer loop
        outer_train_filename = "outer_{}".format(train_filename_temp.format(k))
        outer_test_filename = "outer_{}".format(test_filename_temp.format(k))
        sdw_otest = Chem.SDWriter(outer_test_filename)
        sdw_otrain = Chem.SDWriter(outer_train_filename)
        print("writer outer testset {}".format(k))
        # for m in class_molecules[k]:
        #     pass # write outer testset
        sdw_otest.close()
        trainkeys = class_molecules_keyset.difference(set([k]))
        for l in trainkeys: # outer loop trainsets
            inf = inner_folder.format(l)
            p = pathlib.Path(inf).is_dir()
            if not p:
                os.mkdir(inf)
            inner_train_filename = "{}/inner_{}".format(inf,train_filename_temp.format(l))
            inner_test_filename = "{}/inner_{}".format(inf,test_filename_temp.format(l))
            sdw_itest = Chem.SDWriter(inner_test_filename)
            sdw_itrain = Chem.SDWriter(inner_train_filename)
            print("\twrite trainfile {} using {}".format(outer_train_filename,l))
            pass # writer outer trainset
            sdw_itest.close()

            print("\twrite inner testset {}".format(l))
            # for m in class_molecules[l]:
            #     pass # write inner test

            inner_trainkeys = trainkeys.difference(set([l]))
            for m in inner_trainkeys:
                print("\t\twrite inner_trainfile {} using {}".format(inner_train_filename,m))
            sdw_itrain.close()
        sdw_otrain.close()


def split_to_folds2(filename,label):
    class_molecules = dict()
    class_molecules_keyset = set()
    # class_molecules_keyset.add(0)
    # class_molecules_keyset.add(1)
    # class_molecules_keyset.add(2)
    # class_molecules_keyset.add(3)
    # class_molecules_keyset.add(4)
    # class_molecules = {k:[] for k in class_molecules_keyset}

    sdm = Chem.SDMolSupplier(filename, sanitize = False, removeHs= False, strictParsing=False)
    for mol in sdm:
        mol_label = mol.GetProp(label)
        if not mol_label in class_molecules_keyset:
            class_molecules_keyset.add(mol_label)
            class_molecules[mol_label] = []
        class_molecules[mol_label].append(mol)


    train_filename_temp = "trainset_{}.sdf"
    test_filename_temp = "testset_{}.sdf"
    inner_folder = "trainset_{}"

    for outer_train_ids in itertools.combinations(class_molecules_keyset,len(class_molecules_keyset)-1):
        outer_test = list(class_molecules_keyset.difference(outer_train_ids))
        if len(outer_test)> 1:
            raise AssertionError("to many outer test elements")
        outer_test = outer_test[0]
        inf = inner_folder.format(outer_test)
        p = pathlib.Path(inf).is_dir()
        if not p:
            os.mkdir(inf)
        outer_train_filename = "outer_{}".format(train_filename_temp.format(outer_test))
        outer_test_filename = "outer_{}".format(test_filename_temp.format(outer_test))
        print(outer_test_filename)
        print(outer_train_filename)
        # sdw_otest = Chem.SDWriter(outer_test_filename)
        # sdw_otrain = Chem.SDWriter(outer_train_filename)

        # test = list(class_molecules_keyset.difference(outer_train_ids))

        # test = test[0]
        # print("outer_test: ", test)
        sdw_otest = Chem.SDWriter(outer_test_filename)
        for m in class_molecules[outer_test]:
            sdw_otest.write(m)
        sdw_otest.close()

        print("outer_test: ",outer_test)
        print("outer_train: ", list(outer_train_ids))

        sdw_otrain = Chem.SDWriter(outer_train_filename)
        for ot_id in outer_train_ids:
            for m in class_molecules[ot_id]:
                sdw_otrain.write(m)
        sdw_otrain.close()

        for inner_train_ids in itertools.combinations(outer_train_ids,len(outer_train_ids)-1):
            inner_test = class_molecules_keyset.difference(set([outer_test])).difference(inner_train_ids)
            if len(inner_test) > 1:
                raise AssertionError("to many outer test elements")
            inner_test = list(inner_test)[0]
            print("\n\tinner test: {}".format(inner_test))
            print("\tinner_train: ", list(inner_train_ids))
            inner_train_filename = "{}/{}".format(inf, train_filename_temp.format(inner_test))
            inner_test_filename = "{}/{}".format(inf, test_filename_temp.format(inner_test))

            print(inner_test_filename)
            print(inner_train_filename)
            sdw_itest = Chem.SDWriter(inner_test_filename)
            for m in class_molecules[inner_test]:
                sdw_itest.write(m)
            sdw_itest.close()

            sdw_itrain = Chem.SDWriter(inner_train_filename)
            for inner_tra_id in inner_train_ids:
                for m in class_molecules[inner_tra_id]:
                    sdw_itrain.write(m)
            # print("inner_test: ", inner_test)

            sdw_itrain.close()

if __name__ == '__main__':
    label = sys.argv[1]
    filename = sys.argv[2]
    # split_to_folds(filename,label)
    split_to_folds2(filename, label)

