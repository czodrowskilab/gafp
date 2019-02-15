import os,sys
import rdkit
from rdkit.Chem import SDWriter, MolFromSmiles

activity_label_to_id_map = { "inactive": "0",
                             "active": "1"}


def csv_to_sdf(csv_file, sdf_file, smiles_col, class_col, delim=','):
    sdw = SDWriter(sdf_file)

    with open(csv_file) as fh:
        for i,line in enumerate(fh.readlines()):
            if i == 0:
                continue
            line_split = line.strip().split(delim)
            smiles = line_split[smiles_col].replace('"','')
            act_class = line_split[class_col].replace('"','')
            act_newLabel = activity_label_to_id_map[act_class]
            mol = MolFromSmiles(smiles)
            mol.SetProp("TL", act_newLabel)
            sdw.write(mol)
    sdw.close()


if __name__ == '__main__':
    if(len(sys.argv)<5):

        print("Syntax: python {} <csv_file> <sdf_file> <smiles_col> <class_col>".format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
    csv_to_sdf(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])) # csv_file, sdf_file