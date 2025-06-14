import collections
import time

import math
import numpy as np
import torch

from rl.models import MARLEnv
from gnn.models import HeteroGNN

from load_data import HeteroData, ProcessData
from parsers import pars
from utils.helper import clear_state


def main():
    data = HeteroData(pars.dataset)
    process_data = ProcessData(pars.dataset)

    marl_env = MARLEnv(data, process_data, pars)
    gnn_model = HeteroGNN(data, process_data, pars)

    action_records = collections.deque(maxlen=pars.terminate_tolerance)
    terminate_records = np.zeros(data.n_all_nodes, dtype=bool)

    for episode in range(pars.max_episodes):
        gnn_model.reset(data, process_data, pars)
        curr_chains_with_stop = collections.defaultdict(list)

        for repeat in range(pars.max_repeats + 1):
            marl_env.eval()
            gnn_model.eval()

            with torch.no_grad():
                node_embeddings = gnn_model.attribute_alignment(data.features)
                obs = marl_env.reset(node_embeddings, data.edge_index)

            last_end_node_types = np.full(data.n_all_nodes, -1)
            next_action_masks = data.next_action_masks
            curr_chains = collections.defaultdict(list)
            last_reward = np.zeros(data.n_all_nodes, dtype=pars.numpy_float)

            for timestep in range(pars.max_timesteps):
                with torch.no_grad():
                    curr_action_masks = next_action_masks
                    actions = marl_env.step(obs, curr_action_masks, timestep == repeat == 0, episode)
                    end_node_types = data.action_end_node[actions]

                    curr_raw_chains = [(0,)] * data.n_all_nodes
                    chain_nodes = collections.defaultdict(list)
                    for i, ac in enumerate(actions.tolist()):
                        if ac < data.stop_action_idx:
                            curr_chains[i].append(ac)
                        curr_raw_chains[i] = tuple(curr_chains[i])
                        chain_nodes[curr_raw_chains[i]].append(i)
                        if repeat == pars.max_repeats:
                            curr_chains_with_stop[i].append(ac)
                    chain_nodes = {chain: np.asarray(node_indices) for chain, node_indices in chain_nodes.items()}

                    chain_features = gnn_model.chain_aggregation(node_embeddings, curr_raw_chains, chain_nodes)
                    target_chain_embeddings, target_chain_logits = gnn_model.chain_forward(chain_features[data.target_nodes])

                    next_obs = marl_env.get_next_obs(chain_features, chain_nodes)
                    next_action_masks = marl_env.get_action_masks(chain_nodes)
                    corrects = torch.where(torch.as_tensor(target_chain_logits.argmax(1) == data.target_labels), 1., -1.).cpu().numpy()
                    rewards = marl_env.get_rewards(timestep, corrects, last_reward, end_node_types, curr_raw_chains)
                    dones = np.where(end_node_types < data.stop_node_type, False, True) if timestep < pars.max_timesteps - 1 else np.ones_like(rewards, dtype=bool)

                    continuous_stop_index = data.all_nodes[(last_end_node_types == end_node_types) & (end_node_types == data.stop_node_type)]
                    legal_action_index = np.setdiff1d(data.all_nodes, continuous_stop_index)

                    for agent_idx in range(pars.n_agents):
                        node_idx = np.intersect1d(legal_action_index, data.node_ranges[agent_idx])
                        agent_obs = tuple(np.asarray(o[node_idx].cpu(), dtype=pars.numpy_store_float) for o in obs)
                        agent_actions = actions[node_idx].tolist()
                        agent_next_obs = tuple(np.asarray(o[node_idx].cpu(), dtype=pars.numpy_store_float) for o in next_obs)
                        agent_rewards = rewards[node_idx].tolist()
                        agent_dones = dones[node_idx].tolist()
                        agent_action_mask = np.asarray(curr_action_masks[data.node_ranges[agent_idx]].cpu(), dtype=pars.numpy_store_float)
                        marl_env.replay_buffers[agent_idx].push(agent_obs, agent_actions, agent_rewards, agent_next_obs, agent_dones, agent_action_mask)

                    last_end_node_types = end_node_types
                    last_reward = rewards
                    obs = next_obs

                if episode >= pars.warmup_times:
                    marl_env.train()
                    for _ in range(pars.agent_learn_freq):
                        if pars.agent_joint_train:
                            losses = []
                            for agent_idx in range(pars.n_agents):
                                agent_loss = marl_env.agent_forward(agent_idx)
                                losses.append(pars.agent_joint_ratio[agent_idx] * agent_loss)
                            losses = torch.stack(losses).sum(0)
                            marl_env.learn(losses, marl_env.optimizers)
                        else:
                            for agent_idx in range(pars.n_agents):
                                agent_loss = marl_env.agent_forward(agent_idx)
                                marl_env.learn(agent_loss, marl_env.optimizers[agent_idx])
                        marl_env.soft_update()

                    gnn_model.train()
                    for _ in range(math.ceil(math.log(max(1, episode), math.sqrt(pars.max_timesteps)))):
                        _, logits = gnn_model.chain_forward(chain_features[data.train_index])
                        chain_loss = gnn_model.criterion(logits, data.target_labels[data.train_index])
                        gnn_model.learn(chain_loss, gnn_model.chain_optimizer, name='chain')

                if np.all(dones == True):
                    c_ep = gnn_model.commute_counts(curr_raw_chains)
                    h_ep = np.asarray(target_chain_embeddings.cpu(), dtype=pars.numpy_store_float)
                    pf_ep = corrects[data.mask_index - data.node_ranges[data.target_node_type].start].sum().item()
                    marl_env.episodic_buffer.push(time.time(), pf_ep, c_ep, h_ep)

                    if episode > pars.warmup_times and len(marl_env.episodic_buffer) > pars.episodic_sample_size:
                        hist_chain_adjs, hist_chain_embeddings = marl_env.episodic_buffer.sample(pars.episodic_sample_size, pars.episodic_sampling_ratio)
                        hist_chain_adjs.append(c_ep)

                        gnn_model.train()
                        for _ in range(pars.motif_learn_freq):
                            node_embeddings = gnn_model.attribute_alignment(data.features)
                            chain_features = gnn_model.chain_aggregation(node_embeddings, curr_raw_chains, chain_nodes)
                            motif_features, reg = gnn_model.motif_aggregation(chain_features[data.target_nodes], hist_chain_adjs, hist_chain_embeddings)
                            logits = gnn_model.motif_forward(motif_features[data.train_index], data.features[data.target_node_type][data.train_index])
                            loss = gnn_model.criterion(logits, data.target_labels[data.train_index])
                            gnn_loss = loss + reg
                            gnn_model.learn(gnn_loss, gnn_model.motif_optimizer, name='motif')

                    if repeat == pars.max_repeats:
                        action_records.append(np.asarray(list(curr_chains_with_stop.values())))
                        if episode > pars.warmup_times and len(action_records) >= pars.terminate_tolerance:
                            records = np.asarray(action_records)
                            terminate_records |= np.asarray(records == records[0]).all(0).all(1)
                            if np.all(terminate_records):
                                return

                    clear_state(pars.device)
                    break
