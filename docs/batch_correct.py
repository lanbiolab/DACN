import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm
import DACN

data_root = Path('../data/BaristaSeq/')

proj_list = [
    'Slice_1_removed.h5ad', 'Slice_2_removed.h5ad', 'Slice_3_removed.h5ad'
]

for proj_name in tqdm(proj_list):
    adata_tmp = sc.read(data_root / proj_name)
    adata_tmp.var_names_make_unique()

    adata_tmp.obs['batch_name'] = proj_name.replace("_removed.h5ad", "")
    graph_dict_tmp = DACN.graph_construction(adata_tmp, 12)

    if proj_name == proj_list[0]:
        adata = adata_tmp
        graph_dict = graph_dict_tmp
        name = proj_name
        adata.obs['proj_name'] = proj_name.replace("_removed.h5ad", "")
    else:
        var_names = adata.var_names.intersection(adata_tmp.var_names)
        adata = adata[:, var_names]
        adata_tmp = adata_tmp[:, var_names]
        adata_tmp.obs['proj_name'] = proj_name

        adata = adata.concatenate(adata_tmp)
        graph_dict = DACN.combine_graph_dict(graph_dict, graph_dict_tmp)
        name = name + '_' + proj_name

adata.layers['count'] = adata.X
# sc.pp.filter_genes(adata, min_cells=50)
# sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
# sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
# adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)

from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
# adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
adata.obsm['X_pca'] = adata.X

model = DACN.Dacn(adata.obsm['X_pca'], graph_dict, mode='clustering', device='cuda:0')

# using_dec = True
# if using_dec:
#     model.train_with_dec()
# else:
#     model.train_without_dec()

weight_path = './model_epoch_last.pth'  # 替换为实际的模型权重文件路径
model.load_model(weight_path)

feat, _, _, _ = model.process()
adata.obsm['DACN'] = feat

import harmonypy as hm

meta_data = adata.obs[['batch']]

data_mat = adata.obsm['DACN']
vars_use = ['batch']
ho = hm.run_harmony(data_mat, meta_data, vars_use)

res = pd.DataFrame(ho.Z_corr).T
res_df = pd.DataFrame(data=res.values, columns=['X{}'.format(i+1) for i in range(res.shape[1])], index=adata.obs.index)
adata.obsm[f'Harmony'] = res_df

sc.pp.neighbors(adata, use_rep='Harmony', metric='cosine')
sc.tl.umap(adata, min_dist=0.2)

sc.pl.umap(
    adata,
    color=['batch_name'],
    palette=['#FF7F7F', '#7FFF7F', '#7F7FFF'],
    show=False,
    size=50,
    frameon=False,  # 去掉外边框
    legend_loc='right margin',  # 调整图例位置
    title="DACN"
)

# 获取当前图形并修改散点样式
ax = plt.gca()
for path in ax.spines.values():
    path.set_visible(True)  # 显示边框
    path.set_linewidth(0)   # 设置边框粗细

# 手动调整点的边框颜色
scatter = ax.collections[0]  # 获取散点图
scatter.set_edgecolor('white')  # 设置边框颜色
scatter.set_linewidth(0)  # 设置边框宽度
plt.tight_layout()
plt.savefig('umap_batch.svg', format='svg')
plt.show()

ILISI = hm.compute_lisi(adata.obsm['Harmony'], adata.obs[['batch']], label_colnames=['batch'])[:, 0]
CLISI = hm.compute_lisi(adata.obsm['Harmony'], adata.obs[['Region']], label_colnames=['Region'])[:, 0]

mean_ILISI = np.mean(np.array(ILISI))
mean_CLISI = np.mean(np.array(CLISI))
print("mean_ILISI:", mean_ILISI)
print("mean_CLISI:", mean_CLISI)