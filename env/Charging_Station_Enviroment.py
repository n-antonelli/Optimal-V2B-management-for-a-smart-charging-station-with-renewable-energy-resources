import numpy as np
import pandas as pd
import os
import scipy.io
import sys
import gym
import pathlib
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding
from scipy.io import loadmat, savemat
from env.utils import Init_Values
from env.utils import Simulate_Actions3
import time


class ChargingEnv(gym.Env):
    def __init__(self, price=6, solar=1, reset_flag=0):

        # Parameters
        self.number_of_cars = 10
        self.number_of_days = 1
        self.reset_flag = reset_flag
        self.price_flag = price             # Curva de precio elegida
        self.solar_flag = solar             # Habilitacion del Panel PV
        self.done = False

        self.grid_hist = []                 # Grid_Evol_mem
        self.ev_stored_energy = 0
        self.total_stored_energy = 0        # E_almacenada_total
        self.SOC = []
        self.hist_tse = []                  # Lista_E_Consumida_Total
        self.hist_ese = []                  # Lista_E_Almac_Total
        self.timestep = 0
        self.cost_EV = 0
        self.pv_energy_stored = 0
        self.hist_cost = []                 # Cost_History
        self.pv_res_hist = []
        self.grid_evol = []
        self.res_wasted_evol = []           # res_wasted_evol
        self.penalty_evol = []
        self.BOC = 0

        self.prof_sb = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.51, 0.73, 0.9, 0.98, 1, 0.87, 0.92, 0.9, 0.85,
                               0.76, 0.55, 0.37, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        #self.max_building_cons = 21.5 * 20
        self.max_building_cons = 60.

        self.consume_profile_sb = self.prof_sb * self.max_building_cons


        self.cost_3 = []
        self.total_cost = []

        self.EV_Param = {'charging_effic': 0.91, 'EV_capacity': 30,
                         'discharging_effic': 0.91, 'charging_rate': 11,
                         'discharging_rate': 11}

        # Renewable_Energy
        pv_surface = 2.279 * 1.134 * 40     # = 2,584386 * 40 paneles [m2]

        self.PV_Param = {'PV_Surface': pv_surface, 'PV_effic': 0.21}

        self.current_folder = os.path.realpath(os.path.join(os.path.dirname(__file__), '')) + '\\Files\\'

        low = np.array(np.zeros(3*4+2*self.number_of_cars), dtype=np.float32)     # Lower threshold of state space
        high = np.array(np.ones(3*4+2*self.number_of_cars), dtype=np.float32)     # Upper threshold of state space
        # Definicion de espacio de accion
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.number_of_cars,), dtype=np.float32)
        # Definicion de espacio de estados
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed
        ####################################################################################################
        ####################################################################################################

    def step(self, actions):

        # Cost_EV: Costo por no cargar 100% un auto,
        [reward, grid, en_wasted, self.cost_EV, self.BOC] \
            = Simulate_Actions3.simulate_clever_control(self, actions)

        self.SOC = self.BOC

        # Almacenar datos en variables historicas

        self.grid_hist.append(grid)
        self.grid_evol.append(grid)
        self.res_wasted_evol.append(en_wasted)
        self.penalty_evol.append(self.cost_EV)
        self.hist_cost.append(reward)

        # ------------------------------------------------------------------------------------------------------
        # Guarda el Total_Charging en un vector
        if self.timestep == 0:
            self.hist_tse.clear()
            self.hist_ese.clear()
        self.hist_tse.append(self.total_stored_energy)
        self.hist_ese.append(self.ev_stored_energy)


        # ------------------------------------------------------------------------------------------------------

        self.timestep = self.timestep + 1
        conditions = self.get_obs()

        if self.timestep == 24:
            self.done = True
            self.timestep = 0
            results = {'BOC': self.BOC, 'Grid_Final': self.grid_evol, 'RES_wasted': self.res_wasted_evol,
                       'Penalty_Evol': self.penalty_evol, 'Renewable': self.Energy['Renewable'],
                       'Cost_History': self.hist_cost}

            pv_generation = results['Renewable'][0][:24]
            self.pv_energy_stored = pv_generation - results['RES_wasted']

            # En_consumida_total = E. Consumida de la red + Energía consumida del panel
            tse = results['Grid_Final'] + self.pv_energy_stored

            Energia_desp_EV = np.zeros(len(tse))

            self.cost_3.append(np.mean(self.penalty_evol))
            self.total_cost.append(np.mean(self.hist_cost))
            self.pv_res_hist.append(np.mean(self.pv_energy_stored))

            savemat(self.current_folder + '\Results.mat', {'Results': results})
        ###############################################################################################################

        cost_1 = reward - self.cost_EV
        self.info = {'SOC': self.SOC, 'Presence': self.Invalues['present_cars'], 'Cost1': cost_1}
        return conditions, -reward, self.done, self.info



    def reset(self):
        self.timestep = 0
        self.day = 1
        self.done = False

        # Consumed: Array vacío de igual tamaño de Renewable para guardar energía renovable consumida
        # Renewable: Array de [dias, horas] con energía que genera el panel fotovoltáico
        # Price: Array de [dias, horas] con los precios en cada día del experimento
        # Radiation: Array de [dias, horas] con la radiación solar disponible

        consumed, renewable, price, radiation = self.energy_calculation()
        self.Energy = {'Consumed': consumed, 'Renewable': renewable, 'Price': price, 'Radiation': radiation}
        ######################################################################################

        if self.reset_flag == 0:   # RESETEO

            # ArrivalT: Hora de llegada de cada auto
            # DepartureT: Hora de salida de cada auto
            # evolution_of_cars: Cantidad de autos por hora
            # present_cars: Mapa de qué auto está presente a cada hora

            [BOC, arrival_time, departure_time, evol_of_cars, present_cars] = Init_Values.InitialValues_per_day(self)
            self.Invalues = {'BOC': BOC, 'ArrivalT': arrival_time, 'evolution_of_cars': evol_of_cars,
                             'DepartureT': departure_time, 'present_cars': present_cars}
            savemat(self.current_folder + '\Initial_Values.mat', self.Invalues)

        else:   # NO RESETEO
            contents = loadmat(self.current_folder + '\Initial_Values.mat')
            self.Invalues = {'BOC': contents['BOC'], 'Arrival': contents['ArrivalT'][0],
                             'evolution_of_cars': contents['evolution_of_cars'], 'Departure': contents['DepartureT'][0],
                             'present_cars': contents['present_cars'], 'ArrivalT': [], 'DepartureT': []}

            for ii in range(self.number_of_cars):
                # Guarda todos los valores de Arrival y Departure en ArrivalT y DepartureT
                self.Invalues['ArrivalT'].append(self.Invalues['Arrival'][ii][0].tolist())
                self.Invalues['DepartureT'].append(self.Invalues['Departure'][ii][0].tolist())

        return self.get_obs()

    def get_obs(self):
        if self.timestep == 0:
            self.hist_cost = []
            self.grid_evol = []
            self.res_wasted_evol = []
            self.penalty_evol = []
            self.BOC = self.Invalues["BOC"]

        # leave: Autos que se van
        # Departure_hour: Hora que falta para salir
        [self.leave, Departure_hour, Battery] = self.simulate_station()

        #disturbances = np.array([self.Energy["Radiation"][0, self.timestep] / 1000, self.Energy["Price"][0, self.timestep] / 0.1])
        disturbances = np.array(
            [self.Energy["Radiation"][0, self.timestep] / 1000,
             self.Energy["Price"][0, self.timestep] / 0.1,
             self.consume_profile_sb[self.timestep]])
        # disturbances = [Radiación Actual / 1000, Precio Actual / 0.1] --> para mapear a [0,1]

        #predictions = np.concatenate((np.array([self.Energy["Radiation"][0, self.timestep + 1:self.timestep + 4] / 1000]), np.array([self.Energy["Price"][0, self.timestep + 1:self.timestep + 4] / 0.1])), axis=None)
        predictions = np.concatenate((np.array(
            [self.Energy["Radiation"][0, self.timestep + 1:self.timestep + 4] / 1000]), np.array(
            [self.Energy["Price"][0, self.timestep + 1:self.timestep + 4] / 0.1]), np.array(
            [self.consume_profile_sb])[0, self.timestep + 1:self.timestep + 4]), axis=None)

        # predictions = [Rad+1; Rad+3, Precio+1; Precio+3]

        states = np.concatenate((np.array(Battery), np.array(Departure_hour)/24), axis=None)
        # states = [SoC1; SoC10, Horas para salir1; Horas para salir10]


        observations = np.concatenate((disturbances, predictions, states), axis=None)   # State
        # observations = [disturbances, predictions, states]

        return observations

    def energy_calculation(self):
        days_of_experiment = self.number_of_days
        current_folder = self.current_folder
        price_flag = self.price_flag
        solar_flag = self.solar_flag

        contents = scipy.io.loadmat(current_folder + 'Weather.mat')
        x_forecast = contents['mydata']

        temperature = np.zeros([24 * (days_of_experiment + 1), 1])
        humidity = np.zeros([24 * (days_of_experiment + 1), 1])
        solar_radiation = np.zeros([24 * (days_of_experiment + 1), 1])
        minutes_of_timestep = 60

        count = 0
        for ii in range(0, minutes_of_timestep * 24 * (days_of_experiment + 1), minutes_of_timestep):
            temperature[count, 0] = (np.mean(x_forecast[ii: ii + 59, 0]))
            humidity[count, 0] = (np.mean(x_forecast[ii: ii + 59, 1]))
            solar_radiation[count, 0] = (np.mean(x_forecast[ii: ii + 59, 2]))
            count = count + 1  # Realiza y guarda el promedio de las magnitudes cada una hora.

        experiment_length = days_of_experiment * (60 / minutes_of_timestep) * 24
        renewable = np.zeros([days_of_experiment, int(60 / minutes_of_timestep) * 48])
        radiation = np.zeros([days_of_experiment, int(60 / minutes_of_timestep) * 48])
        count = 0  # Guarda información todos los días c/ media hora

        for ii in range(0, int(days_of_experiment)):
            for jj in range(0, int((60 / minutes_of_timestep) * 48)):
                scaling_PV = self.PV_Param['PV_Surface'] * self.PV_Param['PV_effic'] / 1000
                scaling_sol = 1.5 * 2  # el 2 es para utilizar el doble de paneles
                xx = solar_radiation[count, 0] * scaling_sol * scaling_PV * solar_flag
                # Energía PV = (radiación * factor solar * superficie * efic. PV) / 1000 [w]
                radiation[ii, jj] = solar_radiation[count, 0]  # Radiation = Array de [dias, horas]
                renewable[ii, jj] = xx  # Renewable = Array de [dias, horas]
                count = count + 1

        price_day = []
        # --------------------------------------
        if price_flag == 1:
            price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        elif price_flag == 2:
            price_day = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05,
                                  0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05])
        elif price_flag == 3:
            price_day = np.array([0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                                  0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070])
        elif price_flag == 4:
            price_day = np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                  0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1])
        elif price_flag == 5:
            price_day[1, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
            price_day[2, :] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05,
                               0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05]
            price_day[3, :] = [0.071, 0.060, 0.056, 0.056, 0.056, 0.060, 0.060, 0.060, 0.066, 0.066, 0.076, 0.080,
                               0.080, 0.1, 0.1, 0.076, 0.076, 0.1, 0.082, 0.080, 0.085, 0.079, 0.086, 0.070]
            price_day[4, :] = [0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1]
        elif price_flag == 6:
            price_day = np.array([0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.1,
                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.07, 0.07, 0.06, 0.06])

        # Concatena para obtener 48 valores (cada media hora)
        price_day = np.concatenate([price_day, price_day], axis=0)

        price = np.zeros((days_of_experiment, 48))
        for ii in range(0, days_of_experiment):
            price[ii, :] = price_day  # Guarda los precios en cada día del experimento

        # for ii in range(1,days_of_experiment):
        #   Mixing_functions[ii] = sum(Solar[(ii - 1) * 24 + 1:(ii - 1) * 24 + 24]) / 16

        consumed = np.zeros(np.shape(renewable))
        return consumed, renewable, price, radiation
        # Consumed: Array vacío de igual tamaño de Renewable para guardar energía renovable consumida
        # Renewable: Array de [dias, horas] con energía que genera el panel fotovoltáico
        # Price: Array de [dias, horas] con los precios en cada día del experimento
        # Radiation: Array de [dias, horas] con la radiación solar disponible

    def simulate_station(self):

        BOC = self.BOC
        Arrival = self.Invalues['ArrivalT']
        Departure = self.Invalues['DepartureT']
        present_cars = self.Invalues['present_cars']
        number_of_cars = self.number_of_cars
        day = self.day
        hour = self.timestep

        # cálculo de la hora en que salen los EV
        leave = []
        if hour < 24:
            for car in range(number_of_cars):
                Departure_car = Departure[car]
                if present_cars[car, hour] == 1 and (hour + 1 in Departure_car):
                    leave.append(car)  # Si el auto está y se tiene que ir en la hora siguiente ---> Se agrega a leave

        # calculation of the hour each car is leaving
        Departure_hour = []
        for car in range(number_of_cars):
            if present_cars[car, hour] == 0:
                Departure_hour.append(0)  # Si el auto no está --> no tiene hora de salida
            else:
                for ii in range(len(Departure[car])):

                    if hour < Departure[car][ii]:  # Si la hora que de salida > a la actual
                        Departure_hour.append(Departure[car][ii] - hour)
                        # Se guarda la cantidad de horas que falta para que salga de cada auto
                        break
                        # Departure[car] tiene varias horas de salida para cada auto para un día

        # calculation of the BOC of each car
        Battery = []
        for car in range(number_of_cars):
            Battery.append(BOC[car, hour])  # Guarda en Battery[] los SoC de cada auto
        #############################################################

        return leave, Departure_hour, Battery
        # Leave: Autos que se van
        # Departure_hour: Hora que falta para salir
        # Battery: Soc de cada auto.

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return 0
