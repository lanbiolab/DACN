import scanpy as sc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from pathlib import Path

import DACN
adata = sc.read_h5ad('../data/osmfish/osmfish_remove_excluded.h5ad')
adata.var_names_make_unique()


adata.layers['count'] = adata.X.toarray()
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)

graph_dict = DACN.graph_construction(adata, 12)

random_seed = 2023
DACN.fix_seed(random_seed)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model = DACN.Dacn(adata.X, graph_dict, mode='imputation')

using_dec = True
if using_dec:
    model.train_with_dec()
else:
    model.train_without_dec()

feat, _, _, _ = model.process()
adata.obsm['DACN'] = feat

# reconstruction
de_feat = model.recon()
adata.obsm['de_feat'] = de_feat

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
newcmp = LinearSegmentedColormap.from_list('new', ['#EEEEEE','#FF0000'], N=1000)


list_genes = ['Syt6', 'Rorb', 'Plp1', 'Anln']

for gene in list_genes:
    idx = adata.var.index.tolist().index(gene)
    adata.obs[f'{gene}(denoised)'] = adata.obsm['de_feat'][:, idx]
    adata.obs[f'{gene}(raw)'] = adata.X[:, idx]

fig, axes = plt.subplots(1, len(list_genes), figsize=(4 * (len(list_genes)), 4 * 1))
axes = axes.ravel()

for i in range(len(list_genes)):
    gene = list_genes[i]
    sc.pl.spatial(adata, color=f'{gene}', ax=axes[i], vmax='p99', vmin='p1', alpha_img=0, cmap=newcmp,
                  colorbar_loc=None, size=1.6, spot_size=300, show=False)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout(pad=0.5)  # 自动调整子图间距
# plt.savefig("./osmfish_genes.svg", format="svg", dpi=300, bbox_inches="tight")
plt.show()
plt.close()  # 关闭图形释放内存

fig, axes = plt.subplots(1, len(list_genes), figsize=(4 * (len(list_genes)), 4 * 1))
axes = axes.ravel()
for i in range(len(list_genes)):
    gene = list_genes[i]
    sc.pl.spatial(adata, color=f'{gene}', ax=axes[i], vmax='p99', vmin='p1', alpha_img=0, cmap=newcmp,
                  colorbar_loc=None, size=1.6, spot_size=300, show=False)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.subplots_adjust(wspace=0, hspace=0)