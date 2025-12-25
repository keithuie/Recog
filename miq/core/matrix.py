import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from .channel_params import ChannelParameters

class Matrix:
    def __init__(self, channel_params: List[ChannelParameters], initial_capacity: int = 2000):
        self.channel_params = channel_params
        self._num_channels = len(channel_params)
        self._num_trained_instances = 0
        self._capacity = initial_capacity
        self.matrix = np.zeros((self._capacity, self._num_channels), dtype=np.uint8)
        self._max_possible_raw_score = sum(p.kernel.get_max_score() for p in self.channel_params)

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def num_trained_instances(self) -> int:
        return self._num_trained_instances

    @property
    def capacity(self) -> int:
        return self._capacity

    def _exact_match(self, binned_input: np.ndarray) -> Optional[int]:
        if self._num_trained_instances == 0:
            return None
        valid_data = self.matrix[:self._num_trained_instances]
        matches = np.all(valid_data == binned_input, axis=1)
        indices = np.where(matches)[0]
        return int(indices[0]) if indices.size > 0 else None

    def train_instance(self, binned_input: np.ndarray) -> int:
        existing_idx = self._exact_match(binned_input)
        if existing_idx is not None:
            return existing_idx
        if self._num_trained_instances >= self._capacity:
            self._expand_matrix()
        new_idx = self._num_trained_instances
        self.matrix[new_idx] = binned_input
        self._num_trained_instances += 1
        return new_idx

    def _expand_matrix(self):
        new_capacity = self._capacity * 2
        new_matrix = np.zeros((new_capacity, self._num_channels), dtype=np.uint8)
        new_matrix[:self._capacity] = self.matrix
        self.matrix = new_matrix
        self._capacity = new_capacity

    def get_best_match_score(self, binned_input: np.ndarray) -> float:
        if self._num_trained_instances == 0:
            return 0.0
        total_scores = np.zeros(self._num_trained_instances, dtype=np.float32)
        valid_matrix = self.matrix[:self._num_trained_instances]
        for c in range(self._num_channels):
            col_data = valid_matrix[:, c]
            diffs = np.abs(col_data.astype(np.int32) - int(binned_input[c]))
            kernel = self.channel_params[c].kernel
            profile = np.array(kernel._profile)
            c_scores = np.zeros(self._num_trained_instances, dtype=np.float32)
            valid_mask = diffs < len(profile)
            c_scores[valid_mask] = profile[diffs[valid_mask]]
            total_scores += c_scores
        best_raw_score = np.max(total_scores)
        return float(best_raw_score / self._max_possible_raw_score) if self._max_possible_raw_score > 0 else 0.0

    def get_best_match_with_residuals(self, binned_input: np.ndarray) -> Tuple[float, np.ndarray, int]:
        if self._num_trained_instances == 0:
            return 0.0, np.zeros(self._num_channels), -1
        total_scores = np.zeros(self._num_trained_instances, dtype=np.float32)
        valid_matrix = self.matrix[:self._num_trained_instances]
        for c in range(self._num_channels):
            col_data = valid_matrix[:, c]
            diffs = np.abs(col_data.astype(np.int32) - int(binned_input[c]))
            kernel = self.channel_params[c].kernel
            profile = np.array(kernel._profile)
            c_scores = np.zeros(self._num_trained_instances, dtype=np.float32)
            valid_mask = diffs < len(profile)
            c_scores[valid_mask] = profile[diffs[valid_mask]]
            total_scores += c_scores
        best_idx = int(np.argmax(total_scores))
        best_raw_score = total_scores[best_idx]
        residuals = np.zeros(self._num_channels, dtype=np.float32)
        stored_row = self.matrix[best_idx]
        for c in range(self._num_channels):
            diff = int(binned_input[c]) - int(stored_row[c])
            residuals[c] = self.channel_params[c].kernel.get_score(diff)
        normalized_score = float(best_raw_score / self._max_possible_raw_score) if self._max_possible_raw_score > 0 else 0.0
        return normalized_score, residuals, best_idx

    def save_to_file(self, filepath: str):
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Matrix':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
