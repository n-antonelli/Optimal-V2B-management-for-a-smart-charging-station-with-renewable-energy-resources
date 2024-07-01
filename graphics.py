import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy import loadtxt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

sns.set_theme()

actual_date = datetime.now().date()

Train = False
algoritmo = 'RBC'
fecha_ddpg = '2024-02-20'
fecha_ppo = '2024-02-08'

if Train:
    ### COMPARACION DE REWARD

    rew_curves_DDPG = open(f'curves/Rew_DDPG_{fecha_ddpg}.csv', 'rb')
    rew_curves_PPO = open(f'curves/Rew_PPO_{fecha_ppo}.csv', 'rb')
    data_DDPG = gaussian_filter1d(loadtxt(rew_curves_DDPG, delimiter=","), sigma=5)
    data_PPO = gaussian_filter1d(loadtxt(rew_curves_PPO, delimiter=","), sigma=5)

    plt.figure(figsize=(16, 8))
    plt.plot(data_DDPG, label='DDPG', color='tab:red')
    plt.plot(data_PPO, label='PPO', color='tab:blue')
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.legend(loc="lower right", framealpha=0.7, facecolor='white', fontsize="20")
    plt.xlabel("Training Episodes", fontsize = '20')
    plt.ylabel("Episodic reward", fontsize = '20')
    plt.savefig(f'curves/Reward_comp_{actual_date}.png', dpi=600)
    plt.show()
else:

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
        E_PV_curve = loadtxt(open(f'curves/E_almacenada_PV_ppo.csv', 'rb'), delimiter=",")
        E_tot_curve = loadtxt(open(f'curves/E_almacenada_total_{algoritmo}.csv', 'rb'), delimiter=",")

        if algoritmo == 'ddpg':
            z=1
            #E_tot_curve = [0, *E_tot_curve]
            #E_PV_curve = [0, *E_PV_curve]
            #E_net_curve = E_net_curve[10:]

    # CÁLCULO DE ENERGÍA COMPRADA Y SU COSTO
    En_total = np.sum(E_net_curve)
    print(f"Energía total de {algoritmo}: {En_total}")
    Costo_total = np.sum(price_curve * E_net_curve)
    print(f"Costo total de {algoritmo}: {Costo_total}")
    Costo_prom_compra = Costo_total / En_total
    print(f"Costo promedio de compra de {algoritmo}: {Costo_prom_compra}")

    Costo_por_hora = []
    for a in range(len(E_tot_curve)-1):
        Costo_por_hora.append(E_net_curve[a] * price_curve[a])


    fig, ax1 = plt.subplots(figsize=(16, 8))


#    ax1[0].plot(E_net_curve, color='tab:blue', label='Power grid energy')
#    ax1[0].plot(E_PV_curve, color='tab:green', label='PV energy')
#    ax1[0].plot(ev_consume_curve, color='tab:cyan', label='EV Consume')
#    ax2 = ax1[0].twinx()
#    ax2.set_ylabel('Price [$/kWh]')
#    ax2.plot(price_curve, color='tab:red', label='Price')
#    ax1[0].set_xlabel('Time [h]')
#    ax1[0].set_ylabel('Power grid energy [kWh]')
#    ax1[0].legend(loc="upper left", framealpha=0.7, facecolor='white')
#    ax2.legend(loc="upper right", framealpha=0.7, facecolor='white')

#    ax1[1].plot(Costo_por_hora, color='black', label='Energy cost')
#    ax2 = ax1[1].twinx()
#    ax2.set_ylabel('Price [$/kWh]')
#    ax2.plot(price_curve, color='tab:red', label='Price')
#    ax1[1].set_xlabel('Time [h]')
#    ax1[1].set_ylabel('Energy cost[$]')
#    ax1[1].legend(loc="upper left", framealpha=0.7, facecolor='white')
#    ax2.legend(loc="upper right", framealpha=0.7, facecolor='white')

    #ax1.plot(sb_consume_curve, color='tab:orange', label='SB Demand')
    #ax1.plot(E_tot_curve, color='tab:grey', label='Total Consume')
    ax1.plot(ev_consume_curve, color='tab:cyan', label='EVs energy flow')
    ax1.plot(E_net_curve, color='tab:blue', label='Power grid energy')
    #ax1.plot(Costo_por_hora, color='black', label='Energy cost')
    ax1.plot(E_PV_curve, color='tab:green', label='PV energy')

    ax1.tick_params(axis='y')
    ax1.legend(loc="upper left", framealpha=0.7, facecolor='white', fontsize="20")
    #ax1.set_ylim(top=120)
    ax1.set_xlim([0,24])
    ax1.set_ylim(top=90)
    ax1.set_xticks(np.arange(0, 23, step=4))
    ax1.set_xlabel('Time [h]', fontsize = '20')
    ax1.set_ylabel('Energy [kWh]', fontsize = '20')

    ax1.grid(which='major', color='white', linewidth=2.2)
    ax1.xaxis.set_major_locator(MultipleLocator(4))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.grid(which='minor', color='white', linestyle=':', linewidth = 2)

    plt.tick_params(axis='both', which='major', labelsize=20)


    ax2 = ax1.twinx()
    ax2.set_ylabel('Price [$/kWh]', fontsize = '20')
    ax2.plot(price_curve, color='tab:red', label='Price')
    ax2.tick_params(axis='y')
    ax2.legend(loc="upper right", framealpha=0.7, facecolor='white', fontsize="20")
    ax2.set_ylim(top=0.12)
    ax2.grid(axis='y', visible=False)
    plt.tick_params(axis='both', which='major', labelsize=20)




    #ax3 = ax1.twinx()
    #ax3.spines.right.set_position(("axes", 1.08))
    #ax3.plot(Costo_por_hora, color='black', label='Energy cost')
    #ax3.legend(loc="lower left", framealpha=0.7, facecolor='white')

    #plt.title(algoritmo)
    plt.savefig(f'curves/Energy_comp_{actual_date}_{algoritmo}.png', dpi=600)
    plt.show()


    # GRAFICA DE CARGA
    if algoritmo == 'rbc':
        departure_curve = loadtxt(open(f'curves/Presencia_autos_{algoritmo}.csv', 'rb'), delimiter=",")
        soc_curve = loadtxt(open(f'curves/SOC_{algoritmo}.csv', 'rb'), delimiter=",")
    else:
        departure_curve = loadtxt(open(f'curves/Presencia_autos_{algoritmo}.csv', 'rb'), delimiter=",")
        soc_curve = loadtxt(open(f'curves/SOC_{algoritmo}.csv', 'rb'), delimiter=",")

    departure_curve = np.hstack((departure_curve[:, -1].reshape(-1, 1), departure_curve[:, :-1]))

    # Crear el subplot de 2 filas y 5 columnas
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    k = 0

    # Rellenar cada subgráfico con los datos
    for i in range(2): 
        for j in range(5):
            k += 1
            #axs[i, j].plot(departure_curve[k-1, :], label='departure', color='red')
            axs[i, j].step(range(25), departure_curve[k - 1, :], label='Presence', color='green')
            axs[i, j].plot(soc_curve[k-1, :], label='SoC', color='blue')
            axs[i, j].set_title(f'EV Spot {k}')


            axs[i, j].xaxis.set_major_locator(MultipleLocator(10))
            axs[i, j].xaxis.set_minor_locator(AutoMinorLocator(5))
            axs[i, j].grid(which='minor', color='white', linestyle=':', linewidth=1)

    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    axs[0, 0].set_ylabel('SOC')
    axs[1, 0].set_ylabel('SOC')
    axs[1, 0].set_xlabel('Time [h]')
    axs[1, 1].set_xlabel('Time [h]')
    axs[1, 2].set_xlabel('Time [h]')
    axs[1, 3].set_xlabel('Time [h]')
    axs[1, 4].set_xlabel('Time [h]')

    #plt.title(algoritmo)
    # Ajustar el diseño para evitar superposiciones

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    lines, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2)


    # Mostrar el gráfico
    plt.savefig(f'curves/Charging_{actual_date}_{algoritmo}.png', dpi=600)
    plt.show()

