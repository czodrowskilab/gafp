import sys,os
import rdkit
from rdkit.Chem import ForwardSDMolSupplier, SDWriter, MolFromSmiles, AllChem


def floatify(arg):
    v = None
    try:
        v = float(arg)
    except:
        print("couldn't convert %s to float, try int now" % arg, file=sys.stderr)
        sys.stderr.flush()
        try:
            v = int(v)
        except:
            pass
    return v

def create_class_lambdas(arg):
    class_ranges=arg.strip().split(';')
    class_lambda_dict = {}

    for rule in class_ranges:
        rule_split = rule.strip().split(":")
        rule_class = rule_split[0]
        rule_lambda = eval("lambda x: "+ rule_split[1])
        rule_lambda.__name__ = rule_split[1]
        print("key: %s, lambda: %s" % (rule_class, rule_split[1]), rule_lambda)
        class_lambda_dict[rule_class] = rule_lambda

    return class_lambda_dict

def classify(sdf,label,lambdas):
    new_filename = "%s_class.sdf" % sdf.split('.sdf')[0]
    new_label = label + "_class"
    sdm = ForwardSDMolSupplier(sdf, strictParsing=False, removeHs=False, sanitize=False)
    sdw = SDWriter(new_filename)
    counter = -1
    i=0
    for mol in sdm:
        print(i)
        sys.stdout.flush()
        i += 1
        counter += 1
        if mol is None:
            print("%d rdkit couldn't read molecule" % counter, file=sys.stderr)
            sys.stderr.flush()
            continue
        c = None
        prop = floatify(mol.GetProp(label))
        if prop is None:
            print("couldn't convert %s to float or int...skip" % mol.GetProp(label), file=sys.stderr)
            sys.stderr.flush()
            continue
        for k,l in lambdas.items():
            if l(prop):
                c = k
                print("hit %s"%k);sys.stdout.flush()
                break
        if c is None:
            print("%d no prop range matched '%s' ..skip" % (counter, mol.GetProp(label)), prop,type(prop), file=sys.stderr)
            sys.stderr.flush()
            sys.stdout.flush()
            continue
        mol.SetProp(new_label, c)
        try:
            sdw.write(mol)
        except:
            print("couldn't write mol %d to file, try to build mol from smiles" % i, file=sys.stderr)
            mol = MolFromSmiles(mol.GetProp("SMILES"))
            AllChem.Compute2DCoords(mol)
            mol.SetProp(new_label, c)
            try:
                sdw.write(mol)
            except:
                print("couldn't write mol %d to file...skip" % i, file=sys.stderr)
    sdw.close()


if __name__ == '__main__':
    if(len(sys.argv)<4):
        print("Not enough parameters",file=sys.stderr)
        exit(1)
    sdf = sys.argv[1]
    label = sys.argv[2]
    cutoffs = sys.argv[3]
    lambdas = create_class_lambdas(cutoffs)
    classify(sdf,label,lambdas)
