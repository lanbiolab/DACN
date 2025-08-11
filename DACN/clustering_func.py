import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=2023):

    for res in np.arange(0.2, 2, increment):
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['leiden'].unique()) > n_clusters:
            break
    return res-increment


def leiden(adata, n_clusters, use_rep='DACN', key_added='DACN', random_seed=2023):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.leiden(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added] = adata.obs['leiden']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata


def res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=2023):
    for res in np.arange(0.2, 2, increment):
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['louvain'].unique()) > n_clusters:
            break
    return res-increment

def louvain(adata, n_clusters, use_rep='DACN', key_added='DACN', random_seed=2023):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.louvain(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added] = adata.obs['louvain']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata




def mclust_R(adata, n_clusters, use_rep='DACN', key_added='DACN', random_seed=2023):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import os
    os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.1'
    modelNames = 'EEE'
    # 检查 R.dll 是否存在
    dll_path = os.environ['R_HOME'] + '/bin/x64/R.dll'  # 使用正斜杠
    if os.path.isfile(dll_path):
        print("R.dll 存在:", dll_path)
    else:
        print("R.dll 不存在:", dll_path)
        return None  # 提前返回以避免后续错误

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    print("adata.obsm[use_rep]",adata.obsm[use_rep])
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames)
    print(res)  # 查看 Mclust 结果
    mclust_res = np.array(res[-2])-1

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata