import gym
import numpy as np
import Chargym_Charging_Station

import os
from RBC import RBC
import argparse

from datetime import datetime

fecha_actual = datetime.now().date()
#models_dir = f"models/DDPG-{int(time.time())}"
#logdir = f"logs/DDPG-{int(time.time())}"

#if not os.path.exists(models_dir):
    #os.makedirs(models_dir)

#if not os.path.exists(logdir):
    #os.makedirs(logdir)

