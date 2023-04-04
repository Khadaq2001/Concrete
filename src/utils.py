import torch
import numpy as np
import scanpy as sc
import pandas as pd 
from torch.utils import data
from scipy import sparse

def get_device(i = 1):
    if torch.cuda.device_count() >= i +1:
        return torch.device(f"cuda:{i}")
    else:
        return torch.device("cpu")
    
def anndata_load(file_path):
    """Load anndata, with file_path containing mtx file"""
    adata = sc.read_10x_mtx(file_path, var_names='gene_symbols')
    adata.var_names_make_unique()
    return adata

def anndata_preprocess(adata,
                        min_genes = 200,
                        min_cells = 100,
                        n_top_genes = 5000):
    """Preprocess function"""
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata,target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata

def data_loader(adata, batch_size, shuffle = True):
    if sparse.issparse(adata.X):
        dat = adata.X.A
    else:
        dat = adata.X
    dat = torch.Tensor(dat)
    dataloader = data.DataLoader(dat, batch_size =batch_size, shuffle = shuffle) # type: ignore
    return dataloader