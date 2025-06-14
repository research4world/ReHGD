import collections
import heapq
import itertools
import random

import math
import numpy as np
import torch

from utils.helper import next_power
from utils.quant import get_scale, quantize, dequantize


class ReplayBuffer:
    def __init__(self, agent_idx, pars):
        self.numpy_float = pars.numpy_float
        self.numpy_store_float = pars.numpy_store_float
        self.capacity = pars.replay_capacity[agent_idx]
        self.priority_replay = pars.priority_replay
        self.priority_eps = pars.priority_eps

        self.size = 0
        self.capacity = next_power(self.capacity)
        self.buffer = collections.deque(maxlen=self.capacity)

        if self.priority_replay:
            self.priority_alpha = pars.priority_alpha
            self.priority_beta = pars.priority_beta
            self.max_priority = pars.max_priority

            self.buffer = []
            self.indexes = []
            self.priority_sum = [0.] * (2 * self.capacity)
            self.priority_min = [math.inf] * (2 * self.capacity)
            self.next_idx = 0

    def push(self, obs, actions, rewards, next_obs, dones, next_masks):
        group = list(zip(*obs, actions, rewards, *next_obs, dones, next_masks))

        if not self.priority_replay:
            self.buffer.extend(group)
        else:
            curc_idx = self.next_idx
            if self.capacity - self.next_idx >= len(group):
                if self.capacity > len(self.buffer):
                    self.buffer.extend(group)
                else:
                    self.buffer[self.next_idx: self.next_idx + len(group)] = group
                indexes = range(self.next_idx, self.next_idx + len(group))
            else:
                i = self.capacity - self.next_idx
                j = len(group) - i
                if self.capacity > len(self.buffer):
                    self.buffer.extend(group[: i])
                else:
                    self.buffer[self.next_idx:] = group[: i]
                self.buffer[: j] = group[i: i + min(j, self.capacity)]
                indexes = itertools.chain(range(self.next_idx, self.capacity), range(min(j, self.capacity)))
            self.next_idx = (curc_idx + len(group)) % self.capacity

            priority_alpha = math.pow(self.max_priority, self.priority_alpha)
            for idx in indexes:
                self._set_priority(idx, priority_alpha)

        self.size = min(self.capacity, self.size + len(group))

    def sample(self, batch_size):
        if not self.priority_replay:
            self.indexes = random.sample(range(len(self.buffer)), batch_size)
            weights = np.ones(1, dtype=self.numpy_float)
        else:
            ps = [random.random() * self._sum for _ in range(batch_size)]
            self.indexes = [self.find_prefix_sum_idx(p) for p in ps]
            prob_min = self._min / self._sum
            max_weight = math.pow((prob_min + self.priority_eps) * self.size, -self.priority_beta)
            ps = [self.priority_sum[self.indexes[i] + self.capacity] / self._sum for i in range(batch_size)]
            weights = [math.pow(p * self.size, -self.priority_beta) / max_weight for p in ps]
            weights = np.asarray(weights, dtype=self.numpy_float).reshape(-1, 1)

        transitions = [self.buffer[idx] for idx in self.indexes]

        return transitions, weights

    def update_priorities(self, priorities):
        for idx, priority in zip(self.indexes, priorities):
            if priority > self.max_priority:
                self.max_priority = priority
            priority_alpha = math.pow(priority, self.priority_alpha)

            self._set_priority(idx, priority_alpha)

    def _set_priority(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        self.priority_sum[idx] = priority_alpha

        while idx >= 2:
            idx >>= 1
            l_child = idx << 1
            r_child = l_child + 1

            priority_min_lc, priority_min_rc = self.priority_min[l_child], self.priority_min[r_child]
            if self.priority_eps <= priority_min_lc and self.priority_eps <= priority_min_rc:
                self.priority_min[idx] = self.priority_eps
            elif priority_min_lc <= priority_min_rc:
                self.priority_min[idx] = priority_min_lc
            else:
                self.priority_min[idx] = priority_min_rc
            self.priority_sum[idx] = self.priority_sum[l_child] + self.priority_sum[r_child]

    @property
    def _sum(self):
        return self.priority_sum[1]

    @property
    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            l_child = idx << 1
            r_child = l_child + 1
            if self.priority_sum[l_child] > prefix_sum:
                idx = l_child
            else:
                prefix_sum -= self.priority_sum[l_child]
                idx = r_child

        return idx - self.capacity

    @staticmethod
    def state_quantize(obs, next_obs, next_masks):
        obs1, obs2 = obs
        obs1_scale = get_scale(obs1)
        obs2_scale = get_scale(obs2)
        obs_scale = max(obs1_scale, obs2_scale)
        q_obs1 = quantize(obs1, obs_scale)
        q_obs2 = quantize(obs1, obs_scale)

        next_obs1, next_obs2 = next_obs
        next_obs1_scale = get_scale(next_obs1)
        next_obs2_scale = get_scale(next_obs2)
        next_obs_scale = max(next_obs1_scale, next_obs2_scale)
        q_next_obs1 = quantize(next_obs1, next_obs_scale)
        q_next_obs2 = quantize(next_obs2, next_obs_scale)

        next_masks[np.isinf(next_masks)] = -1
        q_next_masks = next_masks.astype(np.int8)

        q_obs = (q_obs1, q_obs2)
        q_next_obs = (q_next_obs1, q_next_obs2)

        return q_obs, q_next_obs, q_next_masks, obs_scale, next_obs_scale

    def state_dequantize(self, q_obs, q_next_obs, q_next_masks, obs_scale, next_obs_scale):
        q_obs1, q_obs2 = q_obs
        obs1 = dequantize(q_obs1, obs_scale, self.numpy_float)
        obs2 = dequantize(q_obs2, obs_scale, self.numpy_float)

        q_next_obs1, q_next_obs2 = q_next_obs
        next_obs1 = dequantize(q_next_obs1, next_obs_scale, self.numpy_float)
        next_obs2 = dequantize(q_next_obs2, next_obs_scale, self.numpy_float)

        next_masks = q_next_masks.astype(self.numpy_float)
        next_masks[next_masks == -1] = -math.inf

        obs = (obs1, obs2)
        next_obs = (next_obs1, next_obs2)

        return obs, next_obs, next_masks

    def reset(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class EpisodicBuffer:
    def __init__(self, pars):
        self.torch_float = pars.torch_float
        self.capacity = pars.episodic_capacity
        self.rnn_replay = pars.rnn_replay
        self.numpy_float = pars.numpy_float
        self.device = pars.device

        self.buffer = []
        self.score_map = collections.defaultdict(collections.deque)

    def push(self, stamp, pf_ep, c_ep, h_ep):
        if len(self.buffer) >= self.capacity:
            score = -self.buffer.pop(random.randint(0, len(self.buffer) - 1))
            self.score_map[score].popleft()
            if not self.score_map[score]:
                self.score_map.pop(score)

        heapq.heappush(self.buffer, -pf_ep)
        self.score_map[pf_ep].append((stamp, c_ep, h_ep))

    def sample(self, w, p):
        sample_size = round(w * p)
        top_size = round(w * (1 - p))
        designs = []
        scores = []

        for _ in range(top_size):
            score = heapq.heappop(self.buffer)
            designs.extend(self.score_map[-score])
            scores.append(score)

        for score in random.sample(self.buffer, sample_size):
            designs.extend(self.score_map[-score])
        if self.rnn_replay:
            designs.sort()

        self.buffer.extend(scores)
        heapq.heapify(self.buffer)

        _, c_eps, h_eps = list(zip(*designs))

        c_eps = [c.to(self.device) for c in c_eps]
        h_eps = [torch.as_tensor(h, dtype=self.torch_float, device=self.device) for h in h_eps]

        return c_eps, h_eps

    @staticmethod
    def embedding_quantize(emb):
        emb = np.asarray(emb)[np.newaxis, :]
        scale = get_scale(emb)
        q_emb = quantize(emb, scale)

        return q_emb, scale

    def embedding_dequantize(self, q_emb, scale):
        mat = dequantize(q_emb, scale, self.numpy_float)
        mat = mat.squeeze()

        return mat

    def reset(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
