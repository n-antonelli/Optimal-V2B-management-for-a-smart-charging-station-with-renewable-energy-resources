import argparse

import gym
from stable_baselines3.common.env_checker import check_env

from algos.RBC.RBC import RBC
from algos.RL.ddpg.ddpg import DDPG_Agent
from algos.RL.ppo.ppo import PPO_Agent
from algos.RL.ddpg.config.config import Config
from env.Charging_Station_Enviroment import ChargingEnv


def main(args):

    load_date = '2024-02-20'  # Formato: '2023-11-22'

    config = Config.get().main.trainer

    algo = "DDPG"
    mode = "Eval"
    episode_num = 1 # 1 para evaluar una vez con iguales condiciones

    # env = gym.make('ChargingEnv-v0')
    # check_env(env)
    if episode_num == 1:
        res_flag = 1
    else:
        res_flag = 0


    if algo != "RBC":
        if mode == "Train":
            env = ChargingEnv()
            if algo == "DDPG":
                agent = DDPG_Agent(mode, env, load_date)
                agent.train()
                agent.save_models()
            elif algo == "PPO":
                # Maximum number of steps per episode
                args.state_dim = env.observation_space.shape[0]
                args.action_dim = env.action_space.shape[0]
                args.max_action = float(env.action_space.high[0])
                args.max_episode_steps = 200
                agent = PPO_Agent(env, load_date, args)
                agent.train()
                agent.save_models(agent._ppo.actor, agent._ppo.critic, 'model')
            else:
                print("Ingrese un algoritmo existente en el proyecto")



        elif mode == "Eval":
            reward_agent = 0
            for i in range(episode_num):
                env = ChargingEnv(reset_flag=res_flag)
                if algo == "DDPG":
                    agent = DDPG_Agent(mode, env, load_date)
                    agent.load_models()
                else:
                    args.state_dim = env.observation_space.shape[0]
                    args.action_dim = env.action_space.shape[0]
                    args.max_action = float(env.action_space.high[0])
                    args.max_episode_steps = 200
                    agent = PPO_Agent(env, load_date, args)
                    agent.load_models(agent._ppo.actor, agent._ppo.critic, 'model')

                reward_agent += agent.evaluate()
            reward_agent = reward_agent / episode_num
            print(reward_agent)


    else:
        reward_rbc = 0
        for i in range(episode_num):
            env = ChargingEnv(reset_flag=res_flag)     #0 para diferentes días
            rbc_agent = RBC(env)
            reward_rbc += rbc_agent.main(env)
        reward_rbc = reward_rbc / episode_num
        print(reward_rbc)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for ppo-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(480000), help=" Maximum number of training steps")
    parser.add_argument("--evaluation_steps", type=float, default=24,
                        help="Steps for evaluation phase")
    parser.add_argument("--evaluate_freq", type=float, default=24,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=1200, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=128,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="ppo clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="ppo parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    main(args)
