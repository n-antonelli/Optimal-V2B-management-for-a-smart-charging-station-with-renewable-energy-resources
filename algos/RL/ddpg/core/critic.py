import torch
from torch.nn import Linear, Module
import torch.nn.functional as F

from algos.RL.ddpg.config.config import Config
from algos.RL.ddpg.core.net import Net
from algos.RL.ddpg.utils.utils import init_fan_in_uniform

class Critic(Module):
    def __init__(self, observation_dim, action_dim):
        super(Critic, self).__init__()
        
        config = Config.get().ddpg.critic

        self._observation_linear = Linear(observation_dim, config.layers[0])
        self._action_linear = Linear(action_dim, config.layers[0])
        
        init_fan_in_uniform(self._observation_linear.weight)
        init_fan_in_uniform(self._action_linear.weight)

        self._model = Net(config.layers[0] * 2,
                          1,
                          config.layers[1:],
                          config.init_bound,
                          init_fan_in_uniform,
                          None)

    def forward(self, observation, action):
        observation_ = F.relu(self._observation_linear(observation))
        action_ = F.relu(self._action_linear(action))
        return self._model(torch.cat([observation_, action_], dim=1))