import collections
import copy
import gc
import itertools

import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch_scatter as ts

from rl.buffer import ReplayBuffer, EpisodicBuffer


class MARLEnv:
    def __init__(self, data, process_data, pars):
        self.numpy_float = pars.numpy_float
        self.torch_float = pars.torch_float
        self.n_agents = pars.n_agents
        self.action_size = pars.action_size
        self.warmup_times = pars.warmup_times
        self.max_episodes = pars.max_episodes
        self.epsilon_range = pars.epsilon_range
        self.obs_dim = pars.obs_dim
        self.dqn_dim = pars.dqn_dim
        self.bonus_score = pars.bonus_score
        self.lazy_score = pars.lazy_score
        self.punish_score = pars.punish_score
        self.punish_eps = pars.punish_eps
        self.env_sample_size = pars.env_sample_size
        self.eager_stopping = pars.eager_stopping
        self.replay_sample_sizes = pars.replay_sample_sizes
        self.priority_replay = pars.priority_replay
        self.tau = pars.tau
        self.gamma = pars.gamma
        self.dqn_lr = pars.dqn_lr
        self.dqn_wd = pars.dqn_wd
        self.dqn_clip_norm = pars.dqn_clip_norm
        self.device = pars.device

        self.n_all_nodes = data.n_all_nodes
        self.node_ranges = data.node_ranges
        self.target_node_type = data.target_node_type
        self.stop_node_type = data.stop_node_type
        self.mask_index = data.mask_index
        self.node_rel_node = data.node_rel_node
        self.next_rel_masks = data.next_rel_masks
        self.chain_tails = process_data.chain_tails
        self.short_targets = process_data.short_targets

        warmup = np.ones(self.warmup_times, dtype=self.numpy_float)
        decay = np.linspace(*self.epsilon_range, self.max_episodes - self.warmup_times)
        self.eps = np.concat([warmup, decay]).tolist()

        dest_w = nn.Parameter(torch.empty(self.obs_dim, self.dqn_dim))
        self.q_estimates = nn.ModuleList([DoubleDQN(dest_w, agent_idx == self.target_node_type, pars) for agent_idx in range(self.n_agents)])
        self.q_targets = copy.deepcopy(self.q_estimates)
        for param in self.q_targets.parameters():
            param.requires_grad = False

        if pars.agent_joint_train:
            self.optimizers = optim.RMSprop(self.q_estimates.parameters(), lr=self.dqn_lr, weight_decay=self.dqn_wd)
        else:
            self.optimizers = [optim.RMSprop(q_net.parameters(), lr=self.dqn_lr, weight_decay=self.dqn_wd) for q_net in self.q_estimates]

        self.criterion = nn.HuberLoss(reduction='none' if self.priority_replay else 'mean')

        self.replay_buffers = [ReplayBuffer(agent_idx, pars) for agent_idx in range(self.n_agents)]
        self.episodic_buffer = EpisodicBuffer(pars)

    @staticmethod
    def reset(local_embeddings, edge_index):
        rows, cols = edge_index
        neigh_embeddings = ts.scatter_mean(local_embeddings[cols], rows, dim=0)

        return local_embeddings, neigh_embeddings

    def reset_buffers(self):
        for agent_idx in range(self.n_agents):
            self.replay_buffers[agent_idx].reset()
        self.episodic_buffer.reset()
        gc.collect()

    @torch.no_grad()
    def step(self, observations, action_masks, set_out, episode):
        rng = np.random.default_rng()
        actions = []

        for agent_idx in range(self.n_agents):
            mask = action_masks[self.node_ranges[agent_idx]]
            if rng.random() > self.eps[episode]:
                obs = tuple(o[self.node_ranges[agent_idx]] for o in observations)
                action_values = self.q_estimates[agent_idx](obs, set_out) + mask
                action_probs = torch.softmax(action_values, dim=1)
            else:
                action_values = torch.ones_like(mask) + mask
                action_probs = torch.softmax(action_values, dim=1)
            agent_action = torch.distributions.categorical.Categorical(action_probs).sample()
            actions.append(agent_action.cpu().numpy())

        actions = np.concat(actions)

        return actions

    def environ_sampling(self, indices, sizes, rng):
        row_list = [idx[0] for idx in indices]
        col_list = [idx[1] for idx in indices]
        samples = [i for i, size in enumerate(sizes.tolist()) if size > self.env_sample_size]
        sample_indices = rng.integers(sizes[samples][:, None], size=(len(samples), self.env_sample_size))
        for i, sample_idx in enumerate(samples):
            row_list[sample_idx] = row_list[sample_idx][sample_indices[i]]
            col_list[sample_idx] = col_list[sample_idx][sample_indices[i]]

        return row_list, col_list

    def get_action_masks(self, chain_nodes):
        rows, cols = [], []

        for chain, node_indices in chain_nodes.items():
            if chain:
                tail_matrix = self.chain_tails[chain][node_indices - self.node_ranges[self.node_rel_node[chain[0]][0]].start]
                row, col = tail_matrix.nonzero()
                rows.append(node_indices[row])
                cols.append(col + self.node_ranges[self.node_rel_node[chain[-1]][-1]].start)
            else:
                rows.append(node_indices)
                cols.append(np.full_like(node_indices, -1) if self.eager_stopping else node_indices)

        action_masks = ts.scatter_add(self.next_rel_masks[np.hstack(cols)], torch.as_tensor(np.hstack(rows)).to(self.device), dim=0)
        action_masks = torch.where(action_masks, 0., -math.inf)

        return action_masks

    def get_next_obs(self, local_obs, chain_nodes):
        rng = np.random.default_rng()
        rows, cols = [], []

        for chain, node_indices in chain_nodes.items():
            if chain:
                tail_matrix = self.chain_tails[chain][node_indices - self.node_ranges[self.node_rel_node[chain[0]][0]].start]
                sizes = tail_matrix.sum(1)
                indices = np.array_split(np.vstack(tail_matrix.nonzero()), np.cumsum(sizes[:-1]), axis=1)
                row_list, col_list = self.environ_sampling(indices, sizes, rng)
                rows.append(node_indices[np.hstack(row_list)])
                cols.append(np.hstack(col_list) + self.node_ranges[self.node_rel_node[chain[-1]][-1]].start)
            else:
                rows.append(node_indices)
                cols.append(node_indices)

        next_obs = ts.scatter_mean(local_obs[np.hstack(cols)], torch.as_tensor(np.hstack(rows)).to(self.device), dim=0)

        return local_obs, next_obs

    def get_rewards(self, timestep, corrects, last_rewards, end_node_types, chains):
        rewards = np.empty(self.n_all_nodes, dtype=self.numpy_float)
        mask_index = self.mask_index - self.node_ranges[self.target_node_type].start
        rewards[mask_index] = corrects[mask_index]

        lazy_index = np.flatnonzero(end_node_types == self.stop_node_type)
        rewards[lazy_index] = self.punish_score if timestep == 0 else self.lazy_score
        last_wrong_index = np.intersect1d(np.setdiff1d(self.mask_index, lazy_index), np.flatnonzero(last_rewards < 0))
        bonus_index = np.intersect1d(last_wrong_index, np.flatnonzero(rewards > 0))
        rewards[bonus_index] = np.abs(last_rewards[bonus_index]) + self.bonus_score
        punish_index = np.intersect1d(last_wrong_index, np.flatnonzero(rewards < 0))
        rewards[punish_index] += last_rewards[punish_index]
        rewards = self.reward_propagation(rewards, chains, lazy_index)

        return rewards

    def reward_propagation(self, rewards, chains, lazy_index):
        rng = np.random.default_rng()
        rows, cols, vals = [], [], []

        mask_nodes = self.mask_index - self.node_ranges[self.target_node_type].start
        non_mask_nodes = np.setdiff1d(np.arange(self.n_all_nodes), np.union1d(self.mask_index, lazy_index))
        node_mappings = np.full(self.n_all_nodes, -1)
        node_mappings[non_mask_nodes] = np.arange(len(non_mask_nodes))

        chain_nodes = collections.defaultdict(list)
        for node_idx in non_mask_nodes:
            chain_nodes[chains[node_idx]].append(node_idx)

        confidences = np.empty(len(non_mask_nodes), dtype=self.numpy_float)
        chain_lengths = np.empty(len(non_mask_nodes), dtype=self.numpy_float)

        for chain, node_indices in chain_nodes.items():
            node_indices = np.asarray(chain_nodes[chain])
            if chain:
                short_matrix = self.short_targets[chain][node_indices - self.node_ranges[self.node_rel_node[chain[0]][0]].start][:, mask_nodes]
            else:
                short_matrix = self.short_targets[chain][node_indices][:, mask_nodes]

            sizes = np.diff(short_matrix.indptr)
            indices = np.array_split(np.vstack(short_matrix.nonzero()), np.cumsum(sizes[:-1]), axis=1)
            row_list, col_list = self.environ_sampling(indices, sizes, rng)
            row_list, col_list = np.hstack(row_list), np.hstack(col_list)
            rows.append(node_indices[row_list])
            cols.append(col_list)
            vals.append(short_matrix[row_list, col_list])

            node_indices = node_mappings[node_indices]
            confidences[node_indices] = np.bincount(row_list, vals[-1]) / np.bincount(row_list)
            chain_lengths[node_indices] = len(chain)

        vals = np.reciprocal(np.hstack(vals, dtype=self.numpy_float) + self.punish_eps)
        counts = sp.coo_array((vals, (node_mappings[np.hstack(rows)], np.hstack(cols))), shape=(len(non_mask_nodes), len(mask_nodes)))
        non_target_rewards = counts @ rewards[self.mask_index] / counts.sum(1)
        rewards[non_mask_nodes] = np.emath.logn(np.exp(confidences + self.punish_eps), chain_lengths + np.e) * non_target_rewards

        return rewards

    def agent_forward(self, agent_idx):
        samples, weights = self.replay_buffers[agent_idx].sample(self.replay_sample_sizes[agent_idx])
        obs_local, obs_neigh, actions, rewards, next_obs_local, next_obs_neigh, dones, next_masks = list(zip(*samples))

        obs = (torch.as_tensor(np.vstack(obs_local), dtype=self.torch_float, device=self.device),
               torch.as_tensor(np.vstack(obs_neigh), dtype=self.torch_float, device=self.device))
        actions = torch.as_tensor(actions, device=self.device).reshape(-1, 1)
        rewards = torch.as_tensor(rewards, dtype=self.torch_float, device=self.device).reshape(-1, 1)
        next_obs = (torch.as_tensor(np.vstack(next_obs_local), dtype=self.torch_float, device=self.device),
                    torch.as_tensor(np.vstack(next_obs_neigh), dtype=self.torch_float, device=self.device))
        dones = torch.as_tensor(dones, dtype=self.torch_float, device=self.device).reshape(-1, 1)
        next_masks = torch.as_tensor(np.vstack(next_masks), dtype=self.torch_float, device=self.device)
        prior_w = torch.as_tensor(weights, dtype=self.torch_float, device=self.device)

        q_values = self.q_estimates[agent_idx](obs)
        q_values = torch.gather(q_values, dim=-1, index=actions)
        with torch.no_grad():
            q_targets = self.q_targets[agent_idx](next_obs)
            next_q_vals = self.q_estimates[agent_idx](next_obs)
            best_actions = torch.max(next_q_vals + next_masks, dim=1, keepdim=True)[1]
            q_targets = torch.gather(q_targets, dim=1, index=best_actions)
            q_targets = rewards + self.gamma * (1. - dones) * q_targets

        loss = (prior_w * self.criterion(q_values, q_targets)).mean()

        if self.priority_replay:
            new_priorities = torch.abs((q_values.detach() - q_targets).cpu()).squeeze().tolist()
            self.replay_buffers[agent_idx].update_priorities(new_priorities)

        return loss

    def train(self):
        self.q_estimates.train()
        self.q_targets.eval()

    def eval(self):
        self.q_estimates.eval()
        self.q_targets.eval()

    def learn(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_estimates.parameters(), max_norm=self.dqn_clip_norm)
        optimizer.step()

    def soft_update(self):
        for target_param, local_param in zip(self.q_targets.parameters(), self.q_estimates.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)


class DoubleDQN(nn.Module):
    def __init__(self, dest_w, task_driven, pars):
        super(DoubleDQN, self).__init__()
        self.torch_float = pars.torch_float
        self.obs_dim = pars.obs_dim
        self.dqn_dim = pars.dqn_dim
        self.action_size = pars.action_size
        self.feature_dropout = pars.feature_dropout
        self.dqn_dropout = pars.dqn_dropout
        self.n_dqn_layers = pars.n_dqn_layers
        self.noisy_qet = pars.noisy_qet
        self.task_driven = task_driven
        self.device = pars.device

        self.s_w = nn.Parameter(torch.empty(self.obs_dim, self.dqn_dim, dtype=self.torch_float))
        if self.task_driven:
            self.register_parameter('d_w', dest_w)
        else:
            self.register_buffer('d_w', dest_w)

        self.drop = nn.Dropout(self.feature_dropout)

        self.conv = nn.Conv1d(self.dqn_dim, self.dqn_dim, kernel_size=2)
        self.fc = nn.Sequential(
            *list(itertools.chain(*[[
                nn.Linear(self.dqn_dim, self.dqn_dim),
                nn.ReLU(),
                nn.Dropout(self.dqn_dropout)] for _ in range(self.n_dqn_layers)]))
        )

        if self.noisy_qet:
            self.noisy_fc = NoisyLinear(self.dqn_dim, self.dqn_dim, True, pars)
            self.output = NoisyLinear(self.dqn_dim, self.action_size, False, pars)
        else:
            self.register_parameter('noisy_fc', None)
            self.output = nn.Linear(self.dqn_dim, self.action_size, bias=False)

        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        init.kaiming_uniform_(self.s_w, a=math.sqrt(5))
        if self.task_driven:
            init.kaiming_uniform_(self.d_w, a=math.sqrt(5))

    def forward(self, obs, set_out=False, agent_training=True):
        s_obs, d_obs = obs
        s_h = torch.mm(self.drop(s_obs), self.s_w)
        d_h = torch.mm(self.drop(d_obs), self.d_w)
        h = torch.stack([s_h, d_h], dim=-1)
        h = F.relu(self.conv(h)).squeeze()
        h = self.fc(h)

        if self.noisy_qet:
            if set_out:
                self.reset_noise()
            h = F.relu(self.noisy_fc(h, agent_training))
            out = self.output(h, agent_training)
        else:
            out = self.output(h)

        return out

    def reset_noise(self):
        self.noisy_fc.reset_noise()
        self.output.reset_noise()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, pars):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.sigma = pars.noisy_sigma
        self.device = pars.device

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        if self.bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_buffer('bias_epsilon', None)

        self.reset_parameters()
        self.reset_noise()
        self.to(self.device)

    def reset_parameters(self):
        mu_range = 1. / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma / math.sqrt(self.in_features))
        if self.bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma / math.sqrt(self.out_features))

    def forward(self, x, agent_training=True):
        if agent_training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = (self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)) if self.bias else None
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        x = F.linear(x, weight, bias)

        return x

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_out, epsilon_in))
        if self.bias:
            self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())

        return x
