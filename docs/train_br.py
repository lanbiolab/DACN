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
data_root = Path('../data/BRCA1')

# sample name
sample_name = 'V1_Human_Breast_Cancer_Block_A_Section_1'
adata = sc.read_visium(data_root / sample_name)
print(adata)
adata.var_names_make_unique()

df_meta = pd.read_csv(data_root / 'metadata.tsv', sep='\t')
# 确保元数据中的索引与 adata.obs 中的索引匹配
df_meta.set_index('ID', inplace=True)
adata.obs.index = adata.obs.index.astype(str)
df_meta.index = df_meta.index.astype(str)
adata.obs['fine_annot_type'] = df_meta['fine_annot_type']
n_clusters = len(adata.obs['fine_annot_type'].unique())
print(n_clusters)
print(adata.obs.isna().sum())
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

sedr_net = DACN.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
#
# using_dec = True
# if using_dec:
#     sedr_net.train_with_dec(N=1)
# else:
#     sedr_net.train_without_dec(N=1)

weight_path = './model_epoch_last.pth'  # 替换为实际的模型权重文件路径
sedr_net.load_model(weight_path)

sedr_feat, _, _, _ = sedr_net.process()
adata.obsm['DACN'] = sedr_feat
print(sedr_feat.shape)
DACN.mclust_R(adata, n_clusters, use_rep='DACN', key_added='DACN')

# DACN.leiden(adata, n_clusters, use_rep='DACN', key_added='DACN')

sub_adata = adata[~pd.isnull(adata.obs['fine_annot_type'])]
ARI = metrics.adjusted_rand_score(sub_adata.obs['fine_annot_type'], sub_adata.obs['DACN'])
NMI = metrics.normalized_mutual_info_score(sub_adata.obs['fine_annot_type'], sub_adata.obs['DACN'])
print("ARI", ARI)
print(sub_adata.obs['fine_annot_type'])
print(sub_adata.obs['DACN'])
fig, axes = plt.subplots(1,2,figsize=(4*2, 4))
sc.pl.spatial(adata, color='fine_annot_type', ax=axes[0], show=False)
sc.pl.spatial(adata, color='DACN', ax=axes[1], show=False)
axes[0].set_title('Manual Annotation')
axes[1].set_title('ARI=%.4f, NMI=%.4f' % (ARI, NMI))
plt.tight_layout()
output_path = os.path.join('DACN_ARI_NMI', "br.svg")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()