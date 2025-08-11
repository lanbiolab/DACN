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

adata = sc.read_h5ad('../data/MERFISH/MERFISH_0.04.h5ad')
print(adata)
print(adata.obs['Region'])
print(adata.obsm['spatial'])
n_clusters = len(adata.obs['Region'].unique())
print(n_clusters)
adata.obsm['X_pca'] = adata.X
num_genes = adata.X.shape[1]
graph_dict = DACN.graph_construction(adata, 12)
print(graph_dict)
max_gs = 399
mask_ratio = 0.1
mask = np.random.binomial(1, mask_ratio, size=(num_genes, max_gs))

sedr_net = DACN.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)

using_dec = True
if using_dec:
    sedr_net.train_with_dec(N=1)
else:
    sedr_net.train_without_dec(N=1)

# weight_path = 'model_epoch_549.pth'  # 替换为实际的模型权重文件路径
# sedr_net.load_model(weight_path)

sedr_feat, _, _, _ = sedr_net.process()
adata.obsm['DACN'] = sedr_feat
print(sedr_feat.shape)
DACN.mclust_R(adata, n_clusters, use_rep='DACN', key_added='DACN')

sub_adata = adata[~pd.isnull(adata.obs['Region'])]
ARI = metrics.adjusted_rand_score(sub_adata.obs['Region'], sub_adata.obs['DACN'])
NMI = metrics.normalized_mutual_info_score(sub_adata.obs['Region'], sub_adata.obs['DACN'])
print("ARI", ARI)
print(sub_adata.obs['Region'])
print(sub_adata.obs['DACN'])
fig, axes = plt.subplots(1,2,figsize=(4*2, 4))
sc.pl.spatial(adata, color='Region', ax=axes[0], show=False, spot_size=20)
sc.pl.spatial(adata, color='DACN', ax=axes[1], show=False, spot_size=20)
axes[0].set_title('Manual Annotation')
axes[1].set_title('ARI=%.4f, NMI=%.4f' % (ARI, NMI))
plt.tight_layout()
plt.show()