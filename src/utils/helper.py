import datetime
import gc
import random

import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.init as init
import torch_scatter as ts
from torch_geometric.utils import to_edge_index
from torch_geometric.utils.sparse import set_sparse_value


def seed_everything(device):
    seed = datetime.datetime.now().microsecond

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if 'cuda' in device:
        torch.cuda.manual_seed_all(seed)


def clear_state(device):
    clear_cuda_cache(device)
    gc.collect()


def clear_cuda_cache(device):
    if 'cuda' in device:
        torch.cuda.empty_cache()


def next_power(x):
    x = math.ceil(abs(x if x else 1))
    if x > 0 and x & (x - 1) != 0:
        x = math.pow(2, math.ceil(math.log2(x)))

    return int(x)


def activation(name):
    if name is None or name == 'Identity':
        return nn.Identity()
    elif name == 'ReLU':
        return nn.ReLU()
    elif name == 'LeakyReLU':
        return nn.LeakyReLU()
    elif name == 'ELU':
        return nn.ELU()
    elif name == 'PReLU':
        return nn.PReLU()
    elif name == 'SiLU':
        return nn.SiLU()
    else:
        raise NotImplementedError


def random_projection(in_dim, unit, rps, pars):
    dtype = pars.torch_float
    shape = (in_dim, unit)

    def random_project_with_kernel_sparse(s=3., std=1., generator=None):
        probs = torch.rand(shape, generator=generator)
        fills = torch.ones(shape) * math.sqrt(s) * std
        kernel = torch.zeros(shape)
        kernel = torch.where(probs >= (1. - 0.5 / s), fills, kernel)
        kernel = torch.where(probs < (0.5 / s), -fills, kernel)
        return kernel

    def random_project_with_kernel_gaussian():
        std = math.sqrt(2. / (shape[0] + shape[1]))
        kernel = torch.randn(shape) * std
        return kernel

    def random_project_with_kernel_orthogonal():
        kernel = torch.empty(shape)
        init.orthogonal_(kernel)
        return kernel

    def random_project_with_kernel_identity():
        kernel = torch.empty(unit)
        init.ones_(kernel)
        return kernel

    if rps.startswith('sp'):
        sparsity = float(rps.split('_')[1])
        weight = random_project_with_kernel_sparse(s=sparsity)
    elif rps == 'Gaussian':
        weight = random_project_with_kernel_gaussian()
    elif rps == 'Orthogonal':
        weight = random_project_with_kernel_orthogonal()
    elif rps == 'Identity':
        weight = random_project_with_kernel_identity()
    else:
        raise NotImplementedError

    return weight.to(dtype)


def spm_normalizer(spm, eps, factor, num_nodes, symmetry=False):
    (row, col), val = to_edge_index(spm)

    val = torch.where(val < eps, 0, val)
    deg = ts.scatter_add(val, col, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow_(factor)
    deg_inv.masked_fill_(torch.isinf(deg_inv), 0)

    val = val * deg_inv[col]
    if symmetry:
        val = deg_inv[row] * val

    spm = set_sparse_value(spm, val)

    return spm


def get_eigen_value(spm):
    spm = spm.coalesce()
    indices = spm.indices().cpu().numpy()
    values = spm.values().cpu().numpy()
    spm = sp.coo_matrix((values, indices), spm.size())
    eigen_value = sp.linalg.eigsh(spm, 1, return_eigenvectors=False).item()

    return eigen_value
