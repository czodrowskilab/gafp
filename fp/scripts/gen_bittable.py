import operator
from sys import argv

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

input_csv = argv[1]
fp_bits = int(argv[2])
fp_radius = 4
fp_threshold = 0.01


def includeRingMembership(s, n):
    r = ';R]'
    d = ']'
    return r.join([d.join(s.split(d)[:n]), d.join(s.split(d)[n:])])


def includeDegree(s, n, d):
    r = ';D' + str(d) + ']'
    d = ']'
    return r.join([d.join(s.split(d)[:n]), d.join(s.split(d)[n:])])


def writePropsToSmiles(mol, smi, order):
    finalsmi = smi
    for i, a in enumerate(order):
        atom = mol.GetAtomWithIdx(a)
        if atom.IsInRing():
            finalsmi = includeRingMembership(finalsmi, i + 1)
        finalsmi = includeDegree(finalsmi, i + 1, atom.GetDegree())
    return finalsmi


def getSubstructSmi(mol, atomID, radius):
    if radius > 0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
        atomsToUse = []
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env = None
    smi = Chem.MolFragmentToSmiles(mol, atomsToUse, bondsToUse=env, allHsExplicit=True, allBondsExplicit=True,
                                   rootedAtAtom=atomID)
    order = eval(mol.GetProp('_smilesAtomOutputOrder'))
    smi2 = writePropsToSmiles(mol, smi, order)
    return smi, smi2


df = pd.read_csv(input_csv)
mols = [Chem.MolFromSmiles(smiles) for smiles in df.UniSMILES.values]

fps = []
substr = dict.fromkeys(range(fp_bits))
for m in range(len(mols)):
    mol = mols[m]
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_bits, bitInfo=info)
    fps.append(fp)
    for key in info.keys():
        center, rad = info[key][0]
        smi, smi2 = getSubstructSmi(mol, center, rad)
        try:
            if smi2 in substr[key].keys():
                substr[key][smi2] += 1
            else:
                substr[key][smi2] = 1
        except:
            substr[key] = {}
            substr[key][smi2] = 1

substr_counts = [len(x) if x is not None else 0 for _, x in substr.items()]
print('Table column entries')
print(np.mean(substr_counts))

variance = np.var(fps, axis=0)
redundant_bits = {}
filtered_bits = {}

for ix, var in enumerate(variance):
    if var <= fp_threshold:
        redundant_bits[ix] = np.round(var * len(fps))
    else:
        filtered_bits[ix] = np.round(var * len(fps))

print(len(redundant_bits))
print(0)

indices = list(filtered_bits.keys())
print(np.mean(np.array(substr_counts)[indices]))
print(len(filtered_bits))

print('##################################')

fig, ax = plt.subplots(figsize=(5, 8))

# just grab first 5 substructs for visualization
keys = np.arange(0, 5)
width = 0.35
cmaps = ['Oranges', 'Purples', 'Reds', 'Blues', 'Greens']
cmap = cm.get_cmap('Dark2')
uniques = []
draw_structs = []
bottoms = []

for ind in keys:
    try:
        # Use a new colour scheme because each colour is a unique struct
        c_index = 0.5
        cmap = cm.get_cmap(cmaps[ind])
        structs = substr[keys[ind]]
        bottom = 0
        sort = sorted(structs.items(), key=operator.itemgetter(1))
        length = len(sort)
        # Grab the two most common substruct for visualization
        draw_structs.append(sort[-1][0])
        draw_structs.append(sort[-2][0])
        uniques.append(len(np.unique(sort)))

        for i in range(1, length):
            if i > 2:
                col = 'black'
            else:
                col = cmap(c_index)
            ax.bar(ind, sort[length - i][1], width, bottom=bottom, color=col, antialiased=True)
            bottom += sort[length - i][1]
            if col != 'black':
                c_index += 0.3
        bottoms.append(bottom)
    except:
        raise
ax.set_xticks(keys)
ax.set_xticklabels(keys)

plt.ylabel('Absolute substructure frequency')
plt.xlabel('Fingerprint bit')
text_yoffset = 5
text_xoffset = -0.125
for i in keys:
    plt.text(i + text_xoffset, bottoms[i] + text_yoffset, uniques[i], weight='bold')

plt.savefig('substructures.png', dpi=400)

# RDKit throws errors, don't know why
for struct_ix in range(len(draw_structs)):
    Draw.MolToImageFile(Chem.MolFromSmarts(draw_structs[struct_ix]), f'substruct_{struct_ix}.png', kekulize=False)
