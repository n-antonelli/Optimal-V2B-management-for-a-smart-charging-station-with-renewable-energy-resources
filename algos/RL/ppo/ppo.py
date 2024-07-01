from datetime import datetime
import os

import numpy as np
import torch
import time

from algos.RL.ppo.core.normalization import Normalization, RewardScaling
from algos.RL.ppo.core.ppo_continuous import PPO_continuous
from algos.RL.ppo.core.replay_buffer import ReplayBuffer


class PPO_Agent:
    def __init__(self, env, load_date, args):

        self._env = env
        self._load_state = load_date
        self._args = args
        self._date = datetime.now().date()

        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._max_action = float(env.action_space.high[0])
        # self._max_episode_steps = env.max_episode_steps
        self._evaluate_num = 0  # Record the number of evaluations
        self._evaluate_rewards = []  # Record the rewards during the evaluating
        self._total_steps = 0  # Record the total steps during the training

        self._replay_buffer = ReplayBuffer(args)
        self._ppo = PPO_continuous(args)

        self._state_norm = Normalization(shape=self._args.state_dim)  # Trick 2:state normalization
        if self._args.use_reward_norm:  # Trick 3:reward normalization
            self._reward_norm = Normalization(shape=1)
        elif self._args.use_reward_scaling:  # Trick 4:reward scaling
            self._reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    ############################################

    def save_models(self, actor, critic, directory):
        """
        Guarda los modelos del actor y del crítico en el directorio especificado.

        Args:
        - actor: Modelo del actor de PyTorch.
        - critic: Modelo del crítico de PyTorch.
        - directory: Directorio donde se guardarán los modelos.
        - actor_filename: Nombre del archivo para el modelo del actor.
        - critic_filename: Nombre del archivo para el modelo del crítico.
        """
        actor_filename = f'ppo_actor_{self._date}.pth'
        critic_filename = f'ppo_critic_{self._date}.pth'

        if not os.path.exists(directory):
            os.makedirs(directory)

        actor_path = os.path.join(directory, actor_filename)
        critic_path = os.path.join(directory, critic_filename)

        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)

    def load_models(self, actor, critic, directory, actor_filename='ppo_actor.pth', critic_filename='ppo_critic.pth'):
        """
        Carga los modelos del actor y del crítico desde el directorio especificado.

        Args:
        - actor: Modelo del actor de PyTorch.
        - critic: Modelo del crítico de PyTorch.
        - directory: Directorio desde donde se cargarán los modelos.
        - actor_filename: Nombre del archivo para el modelo del actor.
        - critic_filename: Nombre del archivo para el modelo del crítico.
        """
        actor_path = os.path.join(directory, actor_filename)
        critic_path = os.path.join(directory, critic_filename)

        actor.load_state_dict(torch.load(actor_path))
        critic.load_state_dict(torch.load(critic_path))

    def _evaluate_policy(self):
        times = 3
        evaluate_reward = 0
        for _ in range(times):
            s = self._env.reset()
            if self._args.use_state_norm:
                s = self._state_norm(s, update=False)  # During the evaluating,update=False
            done = False
            episode_reward = 0
            while not done:
                a = self._ppo.evaluate(s)  # We use the deterministic policy during the evaluating
                if self._args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * self._args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                s_, r, done, _ = self._env.step(action)
                if self._args.use_state_norm:
                    s_ = self._state_norm(s_, update=False)
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward

        return evaluate_reward / times

    def evaluate(self, reward_scaling=False, reward_norm=False):
        episode_rewards = []

        s = self._env.reset()

        ep_cost1 = 0
        ep_cost3 = 0

        episode_reward = 0
        episode_length = 0
        episode_action = 0

        if self._args.use_state_norm:
            s = self._state_norm(s)
        if self._args.use_reward_scaling:
            reward_scaling.reset()

        for step in range(self._args.evaluation_steps):
            a, a_logprob = self._ppo.choose_action(s)  # Action and the corresponding log probability

            s_, r, done, info = self._env.step(a)

            if self._args.use_state_norm:
                s_ = self._state_norm(s_)
            if self._args.use_reward_norm:
                r = reward_norm(r)
            elif self._args.use_reward_scaling:
                r = reward_scaling(r)

            s = s_

            episode_reward += r
            episode_length += 1
            ep_cost1 += info['Cost1']
            ep_cost3 += self._env.cost_EV

            if done:
                SOC = info['SOC']
                Presence = info['Presence']
                Price = np.array([0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.1,
                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.07, 0.07, 0.06, 0.06])
                print("evaluate_length:{} \t evaluate_reward:{:.2f} \t cost_1:{:.2f}  \t cost_3:{:.2f}\t "
                      .format(episode_length, episode_reward, ep_cost1, ep_cost3))
                np.savetxt("curves/Precio.csv", Price, delimiter=", ", fmt='% s')
                np.savetxt("curves/E_almacenada_red_ppo.csv", self._env.grid_hist, delimiter=", ", fmt='% s')
                np.savetxt("curves/E_almacenada_PV_ppo.csv", self._env.Energy['Renewable'][0][:24], delimiter=", ",
                           fmt='% s')
                np.savetxt("curves/EV_consume_ppo.csv", self._env.hist_ese, delimiter=", ", fmt='% s')
                np.savetxt("curves/sb_energy.csv", self._env.consume_profile_sb, delimiter=", ", fmt='% s')
                np.savetxt("curves/Presencia_autos_ppo.csv", Presence, delimiter=", ", fmt='% s')
                np.savetxt("curves/SOC_ppo.csv", SOC, delimiter=", ", fmt='% s')
                np.savetxt("curves/E_almacenada_total_ppo.csv", self._env.hist_tse, delimiter=", ", fmt='% s')

                reward_eval = episode_reward

                s = self._env.reset()
                episode_reward = 0
                episode_length = 0
        return reward_eval

    def train(self):
        while self._total_steps < self._args.max_train_steps:
            s = self._env.reset()
            ep_reward = 0
            ep_cost_1 = 0
            ep_cost_3 = 0
            if self._args.use_state_norm:
                s = self._state_norm(s)
            if self._args.use_reward_scaling:
                self._reward_scaling.reset()
            episode_steps = 0
            done = False
            while not done:
                episode_steps += 1
                a, a_logprob = self._ppo.choose_action(s)  # Action and the corresponding log probability
                if self._args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * self._args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                s_, r, done, info = self._env.step(action)

                if self._args.use_state_norm:
                    s_ = self._state_norm(s_)
                if self._args.use_reward_norm:
                    r = self._reward_norm(r)
                elif self._args.use_reward_scaling:
                    r = self._reward_scaling(r)

                ep_reward += r
                ep_cost_1 += info['Cost1']
                ep_cost_3 += self._env.cost_EV

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self._args.max_episode_steps:
                    dw = True
                else:
                    dw = False

                # Take the 'action'，but store the original 'a'（especially for Beta）
                self._replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                self._total_steps += 1

                # When the number of transitions in buffer reaches batch_size,then update
                if self._replay_buffer.count == self._args.batch_size:
                    self._ppo.update(self._replay_buffer, self._total_steps)
                    self._replay_buffer.count = 0

                # Evaluate the policy every 'evaluate_freq' steps
                if self._total_steps % self._args.evaluate_freq == 0:
                    self._evaluate_num += 1
                    evaluate_reward = self._evaluate_policy()
                    self._evaluate_rewards.append(evaluate_reward)
                    print("evaluate_length:{} \t evaluate_reward: {:.2f}, ep_reward: {:.2f} \t cost_1: {:.2f}  \t cost_3: {:.2f}\t "
                          .format(self._evaluate_num, evaluate_reward, ep_reward, ep_cost_1, ep_cost_3))
                    #writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
        directory_2 = 'curves'
        if not os.path.exists(directory_2):
            os.makedirs(directory_2)
        np.savetxt(f"curves/Rew_PPO_{self._date}.csv", self._evaluate_rewards, delimiter=", ", fmt='% s')
