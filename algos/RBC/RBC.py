import argparse

import gym
import numpy as np


class RBC:
    def __init__(self, env):
        self._env = env
    def select_action(self, states):
        action=[0,0,0,0,0,0,0,0,0,0]
        for car in range(self.number_of_cars):
            #the departure hour for every spot is placed on the last 10 positions in states vector(10 spots)
            #have in mind that departure time is normalized in [0,1] so if T_leave is within the next 3 hours then
            #action[car]=1, else action[car]=solar_radiation or action[car]={mean value of solar radiation and the
            #predicted one hour radiation}
            if states[18+car]==0: #hora de partir del auto (el Tleave del auto 0 comienza en el índice 18)
                action[car]=0    #si el Tleave es 0 ---> no hacer nada
            elif states[18+car]>0 and states[18+car]<0.16667:
                action[car]=1    #si el Tleave es < 3 ---> cargar el auto
            else:
                # solar ratiation is states[0] and the predictions on ratiation are states[2],states[3],states[4]

                #this case describes that if T_leave> 3 hours, then scenario 1: action is equal to the radiation
                #scenario 2: action is equal to the mean value of current radiation and its next hour prediction

                #scenario 1, current value of radiation
                #action[car]=states[0]

                #scenario 2, mean value of current radiation and one hour ahead
                action[car]=(states[0] + states[3]) / 2   # si el Tleave es > 3 va a cargar dependiendo del promedio de los
                                                          # de la radiación actual y la de la próxima hora
                # states[0] + states[2] no son la irradiancia 0 y 1!!!

        return action


    def main(self, env):

        i = 0
        len_test = 1

        rewards_list = []
        for j in range(len_test):
            state = env.reset()
            done = False
            while not done:
                i += 1
                action = RBC.select_action(env, state)
                next_state, rewards, done, info = env.step(action)
                # print(rewards)
                state = next_state
                rewards_list.append(rewards)

            if done:
                SOC = info['SOC']
                Presence = info['Presence']
                # Gráfico b) Evolución Almacenamiento Energía
                np.savetxt("curves/E_almacenada_red_rbc.csv", self._env.grid_hist, delimiter=", ", fmt='% s')
                np.savetxt("curves/E_almacenada_PV_rbc.csv", self._env.Energy['Renewable'][0][:24], delimiter=", ", fmt='% s')
                np.savetxt("curves/EV_consume_rbc.csv", self._env.hist_ese, delimiter=", ", fmt='% s')
                np.savetxt("curves/E_almacenada_total_rbc.csv", self._env.hist_tse, delimiter=", ", fmt='% s')
                # gráfico c) Perfil de carga
                np.savetxt("curves/Presencia_autos_rbc.csv", Presence, delimiter=", ", fmt='% s')
                np.savetxt("curves/SOC_rbc.csv", SOC, delimiter=", ", fmt='% s')


        final_reward = sum(rewards_list)
        avg_reward = final_reward / len_test
        print(avg_reward)
        return avg_reward

