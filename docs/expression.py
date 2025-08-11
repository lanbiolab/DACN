import matplotlib.colors as clr
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

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
import DACN
import cal

random_seed = 2023
DACN.fix_seed(random_seed)
# gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# path
data_root = Path('../DLPFC')

# sample name
sample_name = '151673'
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


sedr_net = DACN.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
weight_path = './673_ARI/model_epoch_542.pth'  # 替换为实际的模型权重文件路径
sedr_net.load_model(weight_path)
sedr_feat, _, _, _ = sedr_net.process()
adata.obsm['DACN'] = sedr_feat
print(sedr_feat.shape)

DACN.mclust_R(adata, n_clusters, use_rep='DACN', key_added='DACN')

# embedding = adata.obsm['DACN']  # 假设 'DACN' 是 adata.obsm 中存储的表示矩阵
# kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # 初始化 KMeans 模型
# adata.obs['DACN'] = kmeans.fit_predict(embedding)  # 聚类结果添加到 adata.obs['DACN']

adata.obs["x_array"] = adata.obsm['spatial'][:, 0]
adata.obs["y_array"] = adata.obsm['spatial'][:, 1]
print(adata.obs["x_array"])
print(adata.obs["y_array"])

#Use domain 0 as an example
target=1
#Set filtering criterials
min_in_group_fraction=0.8
min_in_out_group_ratio= 1.0
min_fold_change=1.5
#Search radius such that each spot in the target domain has approximately 10 neighbors on average
adj_2d=cal.calculate_adj_matrix(x=adata.obs["x_array"], y=adata.obs["y_array"], histology=False)
print(adj_2d)
start, end= np.quantile(adj_2d[adj_2d!=0],q=0.001), np.quantile(adj_2d[adj_2d!=0],q=0.1)
print(start)
print(end)
r=cal.search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(), x=adata.obs["x_array"] , y=adata.obs["y_array"], pred=adata.obs["DACN"].tolist(), start=start, end=end, num_min=10, num_max=14,  max_run=100)
# r=r*5
print(r)
#Detect neighboring domains
nbr_domians=cal.find_neighbor_clusters(target_cluster=target,
                                   cell_id=adata.obs.index.tolist(),
                                   x=adata.obs["x_array"].tolist(),
                                   y=adata.obs["y_array"].tolist(),
                                   pred=adata.obs["DACN"].tolist(),
                                   radius=r,
                                   ratio=1/2)

nbr_domians=nbr_domians[0:3]
de_genes_info=cal.rank_genes_groups(input_adata=adata,
                                target_cluster=target,
                                nbr_list=nbr_domians,
                                label_col="DACN",
                                adj_nbr=True,
                                log=True)
#Filter genes
de_genes_info=de_genes_info[(de_genes_info["pvals_adj"]<0.05)]
filtered_info=de_genes_info
filtered_info=filtered_info[(filtered_info["pvals_adj"]<0.05) &
                            (filtered_info["in_out_group_ratio"]>=min_in_out_group_ratio) &
                            (filtered_info["in_group_fraction"]>min_in_group_fraction) &
                            (filtered_info["fold_change"]>min_fold_change)]
filtered_info=filtered_info.sort_values(by="in_group_fraction", ascending=False)
filtered_info["target_dmain"]=target
filtered_info["neighbors"]=str(nbr_domians)
print("SVGs for domain ", str(target),":", filtered_info["genes"].tolist())

#Plot refinedspatial domains
color_self = clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370',"#EAE7CC","#FD1593"], N=256)
for g in filtered_info["genes"].tolist():
    # # 选取基因表达数据
    exp_values = adata.X[:, adata.var.index == g].flatten()
    exp_log = np.log1p(exp_values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    exp_normalized = scaler.fit_transform(exp_log.reshape(-1, 1)).flatten()
    adata.obs["exp"] = exp_normalized
    #adata.obs["exp"]=adata.X[:,adata.var.index==g]
    ax=sc.pl.scatter(adata, alpha=1, x="x_array", y="y_array", color="exp", title=g, color_map=color_self, show=False, size=100000/adata.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    output_dir = "./results_673/"
    os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建
    plt.savefig(output_dir + g + ".png", dpi=600)
    plt.close()