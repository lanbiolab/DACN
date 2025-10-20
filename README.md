#Note: DACN is developed based on the Windows system. The configuration methods for the rpy2 package differ between Linux and Windows systems.

DACN is a method that used to analysis spatial transcriptomic with varying resolutions and throughputs. 
DACN is implemented using Python 3.9 and R 4.3. We use the PyTorch framework together with the Scanpy and Anndata packages for bioinformatics analysis. 
The rpy2 package enables interaction between Python and R, and the Mclust package in R is used for clustering.

#Note：It is recommended to execute DACN within a conda virtual environment.

# Step1：Install
```python
conda create -n dacn-env python=3.11.3 -y
conda activate dacn-env
pip install -r requirements.txt
```

# Step2: Running

## The DLPFC dataset is employed as an illustrative example.

```python
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

random_seed = 2025
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


from sklearn.decomposition import PCA 
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
model.train_with_dec(N=1)
```
## Clustering

```python
model = DACN.Dacn(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
weight_path = './model_epoch_last.pth'  # model weight
model.load_model(weight_path)
feat, _, _, _ = model.process()
adata.obsm['DACN'] = feat
print(feat.shape)
print(adata.obsm['DACN'][1:5])
DACN.mclust_R(adata, n_clusters, use_rep='DACN', key_added='DACN')

sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
print(sub_adata.obs['DACN'].shape)
print(sub_adata.obs['layer_guess'].shape)
ARI = metrics.adjusted_rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['DACN'])
NMI = metrics.normalized_mutual_info_score(sub_adata.obs['layer_guess'], sub_adata.obs['DACN'])

fig, ax1 = plt.subplots(1,1,figsize=(4, 4))
sc.pl.spatial(sub_adata, color='layer_guess', ax=ax1, show=False)
ax1.set_title(f'{sample_name}')
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_aspect(1)
plt.tight_layout()
# output_path = os.path.join('DACN_ARI_NMI', "Manual Annotation.svg")
# plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()

fig, ax2 = plt.subplots(1,1,figsize=(4, 4))
sc.pl.spatial(adata, color='DACN', ax=ax2, show=False,legend_loc=None)
ax2.set_title('DACN: ARI=%.2f, NMI=%.2f' % (ARI, NMI))
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_aspect(1)
plt.tight_layout()
# output_path = os.path.join('DACN_ARI_NMI', f"DACN_{sample_name}_ARI.svg")
# plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()

## save clustering results
# output_dir1 = ('./clustering')
# if not os.path.exists(output_dir1):
#     os.makedirs(output_dir1)
# output_path = os.path.join(output_dir1, f"{sample_name}.h5ad")
# adata.write_h5ad(output_path)

# sc.pp.neighbors(adata, use_rep='DACN', metric='cosine')
# sc.tl.umap(adata, min_dist=0.2, spread=0.6)
# fig, axes = plt.subplots(1, 1, figsize=(8, 8))
# sc.pl.umap(adata, color='DACN', ax=axes, show=False,size=60,legend_loc=None)
# axes.set_title('DACN')
# axes.set_aspect(1)
#
# for cluster in adata.obs['layer_guess'].unique():
#     # 计算每个类别的 UMAP 中心点
#     cluster_points = adata[adata.obs['layer_guess'] == cluster].obsm['X_umap']
#     x_mean, y_mean = cluster_points[:, 0].mean(), cluster_points[:, 1].mean()
#
#     # 在中心点处添加标签
#     axes.text(
#         x=x_mean,  # X 坐标
#         y=y_mean,  # Y 坐标
#         s=cluster,  # 标签内容
#         fontsize=10,
#         ha='center',  # 水平对齐
#         va='center',  # 垂直对齐
#         color='black',  # 标签颜色
#         fontweight='bold'  # 标签字体加粗
#     )
#
# axes.set_xlabel('')
# axes.set_ylabel('')
#
# plt.tight_layout()
# output_path = os.path.join('./clustering_results', f"DACN_{sample_name}.svg")
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# plt.show()
