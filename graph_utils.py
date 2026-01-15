import numpy as np
import scipy.sparse as sp
from scipy.sparse import *
import torch

from Params import args


def get_use(behaviors_data, device=None):

    behavior_mats = {}

    binary_mat = (behaviors_data != 0).astype(np.float32)

    behavior_mats['A'] = matrix_to_tensor(normalize_adj(binary_mat), device=device)
    behavior_mats['AT'] = matrix_to_tensor(normalize_adj(binary_mat.T), device=device)
    behavior_mats['A_ori'] = None

    return behavior_mats


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_row = np.power(rowsum + 1e-8, -0.5)
    d_row[np.isinf(d_row)] = 0.
    rowsum_diag = sp.diags(d_row)

    colsum = np.array(adj.sum(0)).flatten()
    d_col = np.power(colsum + 1e-8, -0.5)
    d_col[np.isinf(d_col)] = 0.
    colsum_diag = sp.diags(d_col)

    return rowsum_diag.dot(adj).dot(colsum_diag)


def matrix_to_tensor(cur_matrix, device=None):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()  
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
    values = torch.from_numpy(cur_matrix.data).float()  
    shape = torch.Size(cur_matrix.shape)

    tensor = torch.sparse.FloatTensor(indices, values, shape).coalesce().to(torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
