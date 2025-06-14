import itertools

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.sparse import set_sparse_value

from utils.helper import (
    seed_everything,
    clear_cuda_cache,
    activation,
    random_projection,
    spm_normalizer,
    get_eigen_value
)


class ConvScore:
    def __init__(self, data, pars):
        self.torch_float = pars.torch_float
        self.device = pars.device

        self.n_all_nodes = data.n_all_nodes
        self.node_ranges = data.node_ranges
        self.node_rel_node = data.node_rel_node
        self.edge_index = data.edge_index
        self.adj_matrices = data.adj_matrices

        self.select_cache = {}
        edge_index, edge_weight = gcn_norm(self.edge_index)
        self.spm = torch.sparse_coo_tensor(edge_index, edge_weight, (self.n_all_nodes,) * 2, dtype=self.torch_float, device=self.device)

    def get_score(self, *args):
        _, _, rel = args
        r, c = self.node_rel_node[rel]

        if (r, c) not in self.select_cache:
            rows = torch.as_tensor(self.node_ranges[r], device=self.device)
            cols = torch.as_tensor(self.node_ranges[c], device=self.device)
            score = self.spm.index_select(0, rows).index_select(1, cols).coalesce()
            score = torch.mul(score, self.adj_matrices[rel]).coalesce()
            score_sum = torch.sum(score, dim=0).to_dense()
            score_sum[score_sum == 0.] = 1.
            self.select_cache[(r, c)] = torch.mul(score, score_sum.reciprocal())

        return self.select_cache[(r, c)]


class AttnScore:
    def __init__(self, data, pars):
        self.pre_norm = pars.alignment_norm

        self.node_rel_node = data.node_rel_node
        self.adj_matrices = data.adj_matrices

    def get_score(self, *args):
        x, y, rel = args

        if self.pre_norm:
            x = F.normalize(x)
            y = F.normalize(y)
            score = torch.mm(x, y.mT)
        else:
            score = torch.mm(x, y.mT) / math.sqrt(x.size(-1))

        score = score - score.max(0).values
        score = torch.mul(score, self.adj_matrices[rel]).coalesce()
        score = set_sparse_value(score, score.values().exp())
        score_sum = torch.sum(score, dim=0).to_dense()
        score_sum[score_sum == 0.] = 1.
        score = torch.mul(score, score_sum.reciprocal())

        return score


class HeteroGNN(nn.Module):
    def __init__(self, data, process_data, pars):
        super(HeteroGNN, self).__init__()
        self.n_chain_layers = pars.n_chain_layers
        self.n_motif_layers = pars.n_motif_layers
        self.gnn_chain_dim = pars.gnn_chain_dim
        self.gnn_motif_dim = pars.gnn_motif_dim
        self.chain_fusion_type = pars.chain_fusion_type
        self.node_project_type = pars.node_project_type
        self.edge_project_type = pars.edge_project_type
        self.type_project_grad = pars.type_project_grad
        self.feature_dropout = pars.feature_dropout
        self.gnn_dropout = pars.gnn_dropout
        self.gnn_activate_fn = pars.gnn_activate_fn
        self.alignment_norm = pars.alignment_norm
        self.chain_post_norm = pars.chain_post_norm
        self.gnn_filter_type = pars.gnn_filter_type
        self.gnn_residual = pars.gnn_residual
        self.motif_bias = pars.motif_bias
        self.motif_prune_eps = pars.motif_prune_eps
        self.consistency_reg = pars.consistency_reg
        self.consistency_scale = pars.consistency_scale
        self.norm_eps = pars.norm_eps
        self.gnn_lr = pars.gnn_lr
        self.gnn_wd = pars.gnn_wd
        self.gnn_clip_norm = pars.gnn_clip_norm
        self.device = pars.device

        self.n_all_nodes = data.n_all_nodes
        self.n_target_nodes = data.n_target_nodes
        self.n_target_class = data.n_target_class
        self.n_node_types = data.n_node_types
        self.node_ranges = data.node_ranges
        self.target_node_type = data.target_node_type
        self.node_rel_node = data.node_rel_node
        self.oppo_relation = data.oppo_relation
        self.edge_index = data.edge_index
        self.adj_matrices = data.adj_matrices
        self.feature_dims = data.feature_dims
        self.target_connections = process_data.target_connections

        self.chain_features_dict = {}

        self.node_weights = nn.ParameterList([
            nn.Parameter(random_projection(in_dim, self.gnn_chain_dim, self.node_project_type, pars))
            for nt, in_dim in self.feature_dims.items()
        ])

        self.edge_weights = nn.ParameterList([
            nn.Parameter(random_projection(self.gnn_chain_dim, self.gnn_chain_dim, self.edge_project_type, pars),
                         requires_grad=self.type_project_grad) for _ in self.node_rel_node
        ])

        if self.gnn_filter_type == 'conv':
            gnn_filter = ConvScore(data, pars)
        elif self.gnn_filter_type == 'attn':
            gnn_filter = AttnScore(data, pars)
        else:
            raise NotImplementedError
        self.get_score = gnn_filter.get_score

        self.drop = nn.Dropout(self.feature_dropout)

        self.chain_layers = nn.Sequential(
            nn.Linear(self.gnn_chain_dim, self.gnn_chain_dim),
            *list(itertools.chain(*[[
                activation(self.gnn_activate_fn),
                nn.Dropout(self.gnn_dropout),
                nn.Linear(self.gnn_chain_dim, self.gnn_chain_dim)] for _ in range(self.n_chain_layers - 1)]))
        )

        self.chain_output = nn.Sequential(
            activation(self.gnn_activate_fn),
            nn.Dropout(self.gnn_dropout),
            nn.Linear(self.gnn_chain_dim, self.n_target_class, bias=False)
        )

        if self.chain_fusion_type in ('mean', 'pool'):
            self.fusion_encoder = nn.Linear(self.gnn_chain_dim, self.gnn_chain_dim)
        elif self.chain_fusion_type == 'att':
            self.negative_slope = pars.negative_slope
            self.att_dropout = pars.att_dropout
            self.fusion_encoder = nn.Linear(self.gnn_chain_dim, 1, bias=False)
        elif self.chain_fusion_type == 'rnn':
            self.h_n, self.c_n = self.get_past_rnn
            self.fusion_encoder = nn.LSTM(self.gnn_chain_dim, self.gnn_chain_dim)
        else:
            raise NotImplementedError

        self.conv = nn.Conv1d(self.gnn_chain_dim, self.gnn_chain_dim, 2)
        self.fusion_encoder_fc = nn.Sequential(
            activation(self.gnn_activate_fn),
            nn.Dropout(self.gnn_dropout),
            nn.Linear(self.gnn_chain_dim, self.gnn_motif_dim)
        )

        if self.motif_bias:
            self.motif_bias = nn.Parameter(torch.zeros(self.gnn_motif_dim))
        else:
            self.register_parameter('motif_bias', None)

        self.motif_layers = nn.Sequential(
            *list(itertools.chain(*[[
                nn.Linear(self.gnn_motif_dim, self.gnn_motif_dim),
                activation(self.gnn_activate_fn),
                nn.Dropout(self.gnn_dropout)] for _ in range(self.n_motif_layers - 1)])),
            nn.Linear(self.gnn_motif_dim, self.gnn_motif_dim)
        )

        if self.gnn_residual:
            self.gnn_residual_scale = pars.gnn_residual_scale
            self.gnn_residual = nn.Linear(self.feature_dims[self.target_node_type], self.gnn_motif_dim, bias=False)
        else:
            self.register_parameter('residual', None)

        self.motif_norm = nn.RMSNorm(self.gnn_motif_dim, eps=self.norm_eps)

        self.motif_output = nn.Linear(self.gnn_motif_dim, self.n_target_class, bias=False)

        self.chain_params = [params for name, params in dict(self.named_parameters()).items() if 'chain' in name]
        self.motif_params = [params for name, params in dict(self.named_parameters()).items() if 'chain_output' not in name]
        self.chain_optimizer = optim.AdamW(self.chain_params, lr=self.gnn_lr, weight_decay=self.gnn_wd)
        self.motif_optimizer = optim.AdamW(self.motif_params, lr=self.gnn_lr, weight_decay=self.gnn_wd)

        self.criterion = nn.CrossEntropyLoss()

        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        ...

    @property
    def get_past_rnn(self):
        if hasattr(self, 'h_n'):
            return self.h_n, self.c_n
        return None, None

    def attribute_alignment(self, features):
        self.chain_features_dict.clear()

        embeddings = torch.vstack([torch.mm(x, w) for w, x in zip(self.node_weights, features.values())])
        if self.alignment_norm:
            embeddings = F.normalize(embeddings, dim=-1, eps=self.norm_eps)

        return embeddings

    def chain_aggregation(self, node_embeddings, chains, chain_nodes):

        def update_features(source_features, target_features):
            target_chains = sorted({tuple(c) for c in chains[self.node_ranges[nt].start: self.node_ranges[nt].stop] if c}, key=len)

            for chain in target_chains:
                hidden, i = None, 0
                while i < len(chain):
                    sub_chain = chain[i:]
                    if sub_chain in self.chain_features_dict:
                        hidden = self.chain_features_dict[sub_chain]
                        break
                    i += 1

                j = i - 1
                while j >= 0:
                    relation = self.oppo_relation[chain[j]]
                    child_node_type, parent_node_type = self.node_rel_node[relation]

                    if self.edge_project_type == 'Identity':
                        child_features = torch.mul(source_features[self.node_ranges[child_node_type]], self.edge_weights[relation])
                        parent_features = torch.mul(source_features[self.node_ranges[parent_node_type]], self.edge_weights[relation])
                    else:
                        child_features = torch.mm(source_features[self.node_ranges[child_node_type]], self.edge_weights[relation])
                        parent_features = torch.mm(source_features[self.node_ranges[parent_node_type]], self.edge_weights[relation])
                    hidden = child_features if hidden is None else hidden
                    with torch.no_grad():
                        score = self.get_score(child_features, parent_features, relation)
                    hidden = torch.sparse.mm(score.mT, hidden) + parent_features

                    if nt != self.target_node_type and chain[j:] not in self.chain_features_dict:
                        self.chain_features_dict[chain[j:]] = hidden
                    j -= 1

                target_features[chain_nodes[chain]] = hidden[chain_nodes[chain] - self.node_ranges[nt].start]

            clear_cuda_cache(self.device)

        chain_features = node_embeddings
        if not self.training:
            chain_features = node_embeddings.clone()

        for nt in range(self.n_node_types):
            if nt != self.target_node_type:
                update_features(node_embeddings, chain_features)
        nt = self.target_node_type
        update_features(chain_features, chain_features)

        return chain_features

    def chain_forward(self, h, logits=True):
        h = self.chain_layers(self.drop(h))

        if logits:
            logits = self.chain_output(h)
            if self.chain_post_norm:
                logits = F.normalize(logits, dim=-1, eps=self.norm_eps)
            return h, logits

        return h

    def commute_counts(self, chains):
        indices, weights = [], []

        for node_idx in range(self.n_all_nodes):
            commute = self.target_connections[node_idx].get(chains[node_idx], None)
            if commute is not None:
                indices.append(commute.indices())
                weights.append(commute.values())

        chain_based_subgraph = torch.sparse_coo_tensor(torch.hstack(indices), torch.hstack(weights), (self.n_target_nodes,) * 2, device=self.device)

        return chain_based_subgraph

    def get_motif_adj(self, episodic_chain_subgraph):
        new_motif_adj = motif_adj = spm_normalizer(episodic_chain_subgraph[-1], self.motif_prune_eps, -1, self.n_target_nodes)
        for i in range(len(episodic_chain_subgraph) - 1):
            motif_adj = spm_normalizer(torch.mul(motif_adj, episodic_chain_subgraph[i]), self.motif_prune_eps, -1, self.n_target_nodes)
        motif_adj = (motif_adj + new_motif_adj).coalesce()

        motif_adj = add_self_loops(motif_adj.T, None, 1, self.n_target_nodes)[0]
        motif_adj = spm_normalizer(motif_adj, self.motif_prune_eps, -0.5, self.n_target_nodes, True)

        eigen_value = get_eigen_value(motif_adj)
        motif_adj = add_self_loops(motif_adj, None, -eigen_value / 2.)[0]

        return motif_adj

    def motif_aggregation(self, target_chain_features, episodic_chain_subgraph, episodic_embeddings):
        motif_adj = self.get_motif_adj(episodic_chain_subgraph)

        h_new = self.chain_forward(target_chain_features, logits=False)
        h_eps = torch.stack(episodic_embeddings + [h_new.detach()])
        con_reg = self.consistency_scale * torch.linalg.matrix_norm(h_new - h_eps.mean(0)) if self.consistency_reg else 0.

        if self.chain_fusion_type == 'mean':
            h_eps = self.fusion_encoder(h_eps.mean(0))
        elif self.chain_fusion_type == 'pool':
            h_eps = self.fusion_encoder(h_eps).max(0).values
        elif self.chain_fusion_type == 'att':
            x = F.leaky_relu(h_eps, self.negative_slope)
            alpha = self.fusion_encoder(x)
            alpha = F.softmax(alpha, dim=0)
            alpha = F.dropout(alpha, p=self.att_dropout, training=self.training)
            h_eps = torch.mul(alpha, h_eps).sum(0)
        elif self.chain_fusion_type == 'rnn':
            self.h_n = self.h_n if self.h_n is not None else torch.zeros(1, h_eps.size(1), self.gnn_chain_dim, device=self.device)
            self.c_n = self.c_n if self.c_n is not None else torch.zeros(1, h_eps.size(1), self.gnn_chain_dim, device=self.device)
            self.h_n, self.c_n = self.fusion_encoder(h_eps, (self.h_n.detach(), self.c_n.detach()))[1]
            h_eps = self.h_n[0]
        else:
            raise NotImplementedError

        episodic_embeddings = torch.stack([h_new, h_eps], dim=-1)
        h = self.conv(self.drop(episodic_embeddings)).squeeze()
        h = self.fusion_encoder_fc(h)

        motif_features = torch.sparse.mm(motif_adj, h)
        if self.motif_bias is not None:
            motif_features = motif_features + self.motif_bias

        return motif_features, con_reg

    def motif_forward(self, h, target_features):
        h = self.motif_layers(self.drop(h))

        if self.gnn_residual is not None:
            h = h + self.gnn_residual_scale * self.gnn_residual(target_features)
        h = self.motif_norm(h)

        logits = self.motif_output(h)

        return logits

    def learn(self, loss, optimizer, name):
        optimizer.zero_grad()
        loss.backward()
        params = self.chain_params if name == 'chain' else self.motif_params
        torch.nn.utils.clip_grad_norm_(params, max_norm=self.gnn_clip_norm)
        optimizer.step()

    def reset(self, *args, **kwargs):
        seed_everything(self.device)
        self.__init__(*args, **kwargs)
