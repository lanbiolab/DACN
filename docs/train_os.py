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

adata = sc.read_h5ad('../data/osmfish/osmfish_remove_excluded.h5ad')
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

# using_dec = True
# if using_dec:
#     sedr_net.train_with_dec(N=1)
# else:
#     sedr_net.train_without_dec(N=1)

weight_path = './experience_gan/model_epoch_o1.pth'  # 替换为实际的模型权重文件路径
sedr_net.load_model(weight_path)

sedr_feat, _, _, _ = sedr_net.process()
adata.obsm['DACN'] = sedr_feat
print(sedr_feat.shape)
DACN.mclust_R(adata, n_clusters, use_rep='DACN', key_added='DACN')
# DACN.leiden(adata, n_clusters, use_rep='DACN', key_added='DACN')


sub_adata = adata[~pd.isnull(adata.obs['Region'])]
ARI = metrics.adjusted_rand_score(sub_adata.obs['Region'], sub_adata.obs['DACN'])
NMI = metrics.normalized_mutual_info_score(sub_adata.obs['Region'], sub_adata.obs['DACN'])


fig, ax1 = plt.subplots(1,1,figsize=(4, 4))
sc.pl.spatial(adata, color='Region', ax=ax1, show=False, spot_size=20, size=20, color_map='viridis')
ax1.set_title('osmfish')
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_aspect(1)
plt.tight_layout()
output_path = os.path.join('DACN_ARI_NMI', "osmfish.svg")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

fig, ax2 = plt.subplots(1,1,figsize=(4, 4))
sc.pl.spatial(adata, color='DACN', ax=ax2, show=False, spot_size=20, size=20, color_map='viridis',legend_loc=None)
ax2.set_title('GACN: (ARI=%.2f, NMI=%.2f)' % (ARI, NMI))
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_aspect(1)
plt.tight_layout()
output_path = os.path.join('DACN_ARI_NMI', f"GACN_osmfish_ARI.svg")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()


# sc.pp.neighbors(adata, use_rep='DACN', metric='cosine')
# sc.tl.umap(adata, min_dist=0.2, spread=0.4)
# fig, axes = plt.subplots(1, 1, figsize=(8, 8))
# sc.pl.umap(adata, color='DACN', ax=axes, show=False, size=50,legend_loc=None)
#
# axes.set_title('GACN')
# axes.set_aspect(1)
#
# # 添加类别标签（从 'Region' 列中获取）
# for cluster in adata.obs['DACN'].unique():
#     # 计算每个类别的 UMAP 中心点
#     cluster_points = adata[adata.obs['DACN'] == cluster].obsm['X_umap']
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
# plt.tight_layout()
# output_path = os.path.join('./clustering_results', "GACN_osmfish.svg")
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# plt.show()