from rdkit.Chem import PandasTools
from sys import argv

cluster_data = PandasTools.LoadSDF(argv[1]).cluster
input_data = PandasTools.LoadSDF(argv[2])

input_data['cluster'] = cluster_data

PandasTools.WriteSDF(input_data, 'data_cluster_merged.sdf', properties=list(input_data.columns))
