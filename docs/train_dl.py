import numpy as np
import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import DACN

random_seed = 2023
DACN.fix_seed(random_seed)
# gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# path
data_root = Path('../data/DLPFC')

# sample name
sample_name = '151674'
n_clusters = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7
adata = sc.read_visium(data_root / sample_name)
print(adata)
adata.var_names_make_unique()

df_meta = pd.read_csv(data_root / sample_name / 'metadata.tsv', sep='\t')
adata.obs['layer_guess'] = df_meta['layer_guess']
adata.layers['count'] = adata.X.toarray()
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)


from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
print(adata_X.shape[1])
adata.obsm['X_pca'] = adata_X
num_genes = adata_X.shape[1]
graph_dict = DACN.graph_construction(adata, 12)
print(graph_dict)
max_gs = 399
mask_ratio = 0.1
mask = np.random.binomial(1, mask_ratio, size=(num_genes, max_gs))

model = DACN.Dacn(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
using_dec = True
if using_dec:
    model.train_with_dec(N=1)
else:
    model.train_without_dec(N=1)
