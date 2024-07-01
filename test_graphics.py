import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy import loadtxt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

sns.set_theme()

actual_date = datetime.now().date()


algoritmo = 'rbc'
fecha_ddpg = '2024-02-21'
fecha_ppo = '2024-02-19'

if True:

    ### GRAFICA DE GENERACION RBC
    if algoritmo == 'rbc':

        price_curve = loadtxt(open('curves/Precio.csv', 'rb'), delimiter=",")
        sb_consume_curve = loadtxt(open('curves/sb_energy.csv', 'rb'), delimiter=",")
        ev_consume_curve = loadtxt(open(f'curves/EV_consume_{algoritmo}.csv', 'rb'), delimiter=",")
        E_net_curve = loadtxt(open(f'curves/E_almacenada_red_{algoritmo}.csv', 'rb'), delimiter=",")
        E_PV_curve = loadtxt(open(f'curves/E_almacenada_PV_{algoritmo}.csv', 'rb'), delimiter=",")
        E_tot_curve = loadtxt(open(f'curves/E_almacenada_total_{algoritmo}.csv', 'rb'), delimiter=",")

    else:
        ### GRAFICA DE GENERACION DDPG Y PPO
        price_curve = loadtxt(open('curves/Precio.csv', 'rb'), delimiter=",")
        sb_consume_curve = loadtxt(open('curves/sb_energy.csv', 'rb'), delimiter=",")
        ev_consume_curve = loadtxt(open(f'curves/EV_consume_{algoritmo}.csv', 'rb'), delimiter=",")
        E_net_curve = loadtxt(open(f'curves/E_almacenada_red_{algoritmo}.csv', 'rb'), delimiter=",")
        E_PV_curve = loadtxt(open(f'curves/E_almacenada_PV_{algoritmo}.csv', 'rb'), delimiter=",")
        E_tot_curve = loadtxt(open(f'curves/E_almacenada_total_{algoritmo}.csv', 'rb'), delimiter=",")

        if algoritmo == 'ddpg':
            z = 1
            #E_tot_curve = [0, *E_tot_curve]
            #E_PV_curve = [0, *E_PV_curve]
            #E_net_curve = E_net_curve[10:]

    # CÁLCULO DE ENERGÍA COMPRADA Y SU COSTO
    if algoritmo == "ddpg":

        En_total = np.sum(E_net_curve[:-1])
        print(f"Energía total de {algoritmo}: {En_total}")
        Costo_total = np.sum(price_curve*E_net_curve)
        print(f"Costo total de {algoritmo}: {Costo_total}")
    else:
        En_total = np.sum(E_net_curve)
        print(f"Energía total de {algoritmo}: {En_total}")
        Costo_total = np.sum(price_curve * E_net_curve)
        print(f"Costo total de {algoritmo}: {Costo_total}")


    # Cálculo y gráfico de composición de consumo total
    sb_consume_curve = sb_consume_curve[:-4]
    e_EV_curve = []
    Total = []
    Total_sin_EV = []
    for a in range(len(E_tot_curve)):

        if ev_consume_curve[a] < 0:     # si sobra energía a las baterías, se guarda en vector
            e_EV_curve.append(-ev_consume_curve[a])
        else:
            e_EV_curve.append(0)
        # PARA GRÁFICOS ---- Totales de cantidad
        Total.append(E_net_curve[a] + E_PV_curve[a] + e_EV_curve[a])
        Total_sin_EV.append(E_net_curve[a] + E_PV_curve[a])


    def add_value_label(x_list, y_list, y_list_ant, color, altura):
        y_list_ant = np.array(y_list_ant).astype(int)
        myArray = np.array(y_list).astype(int)
        for i in range(0, len(x_list)):
            # para evitar escribir dos veces el mismo número en el mismo lugar
            if myArray[i] != y_list_ant[i]:
                plt.text(i, myArray[i]//altura + 2, myArray[i], ha = "center", fontsize = 20, color = 'black', va = "center")

    index = np.arange(len(sb_consume_curve))
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.bar(index, Total, color='tab:cyan', label = 'Useful EV energy')
    add_value_label(index, E_PV_curve, np.zeros(len(E_PV_curve)), 'tab:cyan', 2)
    ax.bar(index, Total_sin_EV, color='tab:blue', label = 'Useful grid energy')
    add_value_label(index, np.array(Total_sin_EV), E_PV_curve, 'tab:blue', 1)
    ax.bar(index, E_PV_curve, color='tab:green', label = 'Useful PV energy')
    add_value_label(index, np.array(Total), np.array(Total_sin_EV), 'tab:green', 1)
    ax.plot(sb_consume_curve, color='tab:orange', label='SB Demand')
    #ax.plot(E_PV_curve, color='tab:green', label='PV energy')
    #ax.plot(price_curve, color='tab:red', label='Price')
    #ax.plot(sb_consume_curve, color='tab:orange', label='SB Demand')
    ax.plot(E_tot_curve, color='tab:grey', label='Total Consume')
    ax.plot(ev_consume_curve, color='tab:cyan', label='EVs energy flow')
    #ax.plot(E_net_curve, color='tab:blue', label='Power grid energy')

    ax.tick_params(axis='y')
    ax.legend(loc="upper left", framealpha=0.7, facecolor='white', fontsize="20")
    ax.set_xlabel('Time [h]', fontsize = '20')
    ax.set_ylabel('Energy [KWh]', fontsize = '20')
    ax.set_xticks(np.arange(0, 23, step=4))
    ax.set_xlim([-1,24])
    ax.set_ylim(top=130)
    plt.tick_params(axis='both', which='major', labelsize=20)



    ax1 = ax.twinx()
    ax1.set_ylabel('Cost [$]', fontsize = '20')
    ax1.plot(price_curve, color='tab:red', label='Price', linewidth=0.9)
    ax1.tick_params(axis='y')
    ax1.legend(loc="upper right", framealpha=0.7, facecolor='white', fontsize="20")
    ax1.set_ylim(top=0.12)
    ax1.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=20)

    #plt.title(algoritmo)

    plt.savefig(f'curves/Comsume_perce_{actual_date}_{algoritmo}.png', dpi=600)
    plt.show()


