import numpy as np
from numpy import random
import scipy.io


def InitialValues_per_day(self):
    number_of_cars = self.number_of_cars
    ArrivalT = []
    DepartureT = []
    BOC = np.zeros([number_of_cars, 25])
    present_cars = np.zeros([number_of_cars, 25])
    #initial state stochastic creation

    for car in range(number_of_cars):
        present = 0
        pointer = 0
        Arrival_car = []
        Departure_car = []
##############################################################

        for hour in range(24):

            if present == 0:
                arrival = round(random.rand()-0.1)
                if arrival == 1 and hour <= 18:     # El auto puede llegar hasta las 20 hs
                    ran = random.randint(20, 80)    # Tiene una carga de entre 20% y 50% ---> Cambiado de 20% y 80%
                    BOC[car, hour] = ran / 100
                    pointer = pointer+1
                    Arrival_car.append(hour)
                    upper_limit = min(hour + 10, 25)
                    #Departure_car.append(random.randint(hour+4,int(upper_limit)))
                    Departure_car.append(random.randint(hour + 6, int(upper_limit)))
                    # El auto se queda desde 4 hasta 10 hs, siempre dentro del mismo día
            ###########################################################################################################

            if arrival == 1 and pointer > 0:    # Pone 1 en present_cars[car,hour] si ese auto está presente a esa hora
                if hour < Departure_car[pointer-1]:
                    present = 1
                    present_cars[car,hour] = 1
                else:
                    present = 0
                    present_cars[car, hour] = 0
            else:
                present = 0
                present_cars[car, hour] = 0

        ArrivalT.append(Arrival_car)        # Guarda las horas en que llegó cada auto
        DepartureT.append(Departure_car)        # Guarda las horas en que se fue cada auto

    #information vector creator
    evolution_of_cars=np.zeros([24])
    for hour in range(24):
        evolution_of_cars[hour]=np.sum(present_cars[:,hour])
        # Muestra la cantidad de autos que hay en cada hora del día
    ##################################################################

    return BOC, ArrivalT, DepartureT, evolution_of_cars, present_cars
    # BOC: SoC
    # ArrivalT: Hora de llegada de cada auto
    # DepartureT: Hora de salida de cada auto
    # evolution_of_cars: Cantidad de autos por hora
    # present_cars: Mapa de qué auto está presente a cada hora