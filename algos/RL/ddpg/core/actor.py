import torch

from algos.RL.ddpg.config.config import Config
from algos.RL.ddpg.core.net import Net
from algos.RL.ddpg.utils.utils import init_fan_in_uniform


class Actor(Net):
    def __init__(self, observation_dim, action_dim):
        config = Config.get().ddpg.actor

        super(Actor, self).__init__(observation_dim,
                                    action_dim,
                                    config.layers,
                                    config.init_bound,
                                    init_fan_in_uniform,
                                    torch.tanh)
