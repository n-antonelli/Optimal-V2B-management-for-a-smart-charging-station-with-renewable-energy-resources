import numpy as np
import matplotlib.pyplot as plt
import time


def simulate_clever_control(self, actions):
    hour = self.timestep
    consumed = self.Energy['Consumed']
    renewable = self.Energy['Renewable']
    present_cars = self.Invalues['present_cars']
    leave = self.leave
    BOC = self.BOC      # SOC
    perfil_en_cons = self.consume_profile_sb
    #max_building_cons = 21.5 * 20  # máximo consumo diario * cantidad de hogares

    P_charging = np.zeros(self.number_of_cars)

    ###########################################################################
    # Calculation of demand based on actions
    # Calculation of actions for cars
    # ----------------------------------------------------------------------------
    for car in range(self.number_of_cars):
        if actions[car] >= 0:      # Si hay que cargar el auto --> Ec (2) del paper (MaxEnergy)
            max_charging_energy = min([10, (1-BOC[car, hour])*self.EV_Param['EV_capacity']])    # Con el 'min' chequea la Ec (3) del paper
        else:
            max_charging_energy = min([10, BOC[car, hour] * self.EV_Param['EV_capacity']])
            # if action=[-1,1] P_charging[car] = 100*actions[car]/100*max_charging_energy

        if present_cars[car, hour] == 1:      # Si el auto está --> Ec (4) del paper (Pdem)
            # Divide en vez de multiplicar como en el paper
            # P_charging[car] = 100*actions[car]/100*max_charging_energy
            P_charging[car] = actions[car] * max_charging_energy
            # P_charging[car] > 0 ----> Charging
            # P_charging[car] < 0 ----> Discharging
        else:
            P_charging[car] = 0

    # Calculation of next state of Battery based on actions
    # ----------------------------------------------------------------------------
    for car in range(self.number_of_cars):
        if present_cars[car, hour] == 1:      # Si el auto está --> SoC próximo = SoC actual + Pdem/capacidad
            BOC[car, hour+1] = BOC[car, hour] + P_charging[car]/self.EV_Param['EV_capacity']
            # Pdem/capacidad es lo que se va a cargar la batería en la próxima hora

    # Calculation of load electricity consumption
    # ----------------------------------------------------------------------------
    building_consume = perfil_en_cons[hour]

    # Calculation of energy utilization from the PV
    # Calculation of energy coming from Grid
    # ----------------------------------------------------------------------------
    # RES_avail = max([0, Renewable[0, hour] - Consumed[0, hour]])                      # Siempre usa el día 0!!!

    EV_charging = sum(P_charging)                             # Potencia demandada y consumida por todos los autos

    Total_charging = sum(P_charging) + building_consume            # Agregado al Total_charging el consumo del edificio

    self.ev_stored_energy = EV_charging
    self.total_stored_energy = Total_charging

    ##############################################################################
    # First Cost index
    # ----------------------------------------------------------------------------
    # Grid_final = max([Total_charging - RES_avail, 0])      # Lo que se consume de la red
    RES_Gen = max([0, renewable[0, hour]])

    if Total_charging >= 0:      # Sobrante de energía PV y el consumo de la red si los autos consumieron erengía
        RES_wasted = max([RES_Gen - Total_charging, 0])      # Solo hay energía sobrante si generó más
        EV_Wasted = 0
    else:
        RES_wasted = renewable[0, hour]
        EV_Wasted = - Total_charging                        # Remanente de energía de los autos

    En_wasted = RES_wasted + EV_Wasted

    Grid_final = max([Total_charging - RES_Gen, 0])             # Lo que se consume de la red
    Cost_1 = Grid_final*self.Energy["Price"][0, hour]           # Costo por consumo de la red (positivo)
    # Cost_4 = building_consume*self.Energy["Price"][0, hour]     # Costo por consumo de edificio

    # Second Cost index
    # Penalty of wasted RES energy
    # This is not used in this environment version
    # ----------------------------------------------------------------------------
    # RES_avail = max([RES_avail-Total_charging, 0])
    # Cost_2 = -RES_avail * (self.Energy["Price"][0, hour]/2)

    # Third Cost index
    # Penalty of not fully charging the cars that leave
    # ----------------------------------------------------------------------------
    Cost_EV = []
    for ii in range(len(leave)):
        Cost_EV.append(((1-BOC[leave[ii], hour+1])*2)**2)       # Simulate_Station crea leave
        # BOC[leave[ii], hour+1] solo tiene en cuenta el SoC de los autos que se van a ir en la próxima hora
    Cost_3 = sum(Cost_EV)

    #Cost = Cost_1 + Cost_3 + Cost_4
    Cost = Cost_1 * (0.4) + Cost_3 * (0.6)

    return Cost, Grid_final, En_wasted, Cost_3, BOC
    # Cost: Costo total
    # Grid_final: Lo que se consume de la red
    # Cost_3: Costo por no cargar 100% un auto,