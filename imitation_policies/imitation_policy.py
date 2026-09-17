# Abstract class for any imitation policy.

from abc import ABC, abstractmethod
import torch
import numpy as np

class Transition:
    def __init__(self, state, action, next_state, done, log_prob=None, reward=None):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.done = done

        self.log_prob = log_prob
        self.reward = reward
    
    @staticmethod
    def to_tensor(batch, device):
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(device)
        actions = torch.LongTensor(np.array([t.action for t in batch])).to(device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(device)
        dones = torch.FloatTensor(np.array([t.done for t in batch])).to(device)
        log_probs = torch.FloatTensor(np.array([t.log_prob for t in batch if t.log_prob is not None])).to(device) if any(t.log_prob is not None for t in batch) else None
        rewards = torch.FloatTensor(np.array([t.reward for t in batch if t.reward is not None])).to(device) if any(t.reward is not None for t in batch) else None
        return states, actions, next_states, dones, log_probs, rewards
    
class ImitationPolicyAgent(ABC):
    @abstractmethod
    def predict(self, state, deterministic: bool = False):
        """Returns the action to take given a state."""
        pass

    @abstractmethod
    def update_representation(self, expert_batch: list[Transition], learner_batch: list[Transition]):
        """Update the policy's internal representation based on the expert data."""
        pass

    @abstractmethod
    def update_policy(self, learner_batch: list[Transition]):
        """Update the policy's parameters based on the expert data."""
        pass

    @abstractmethod
    def save(self):
        """Save the policy. The path is assumed to be defined internally."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the policy from a given path."""
        pass