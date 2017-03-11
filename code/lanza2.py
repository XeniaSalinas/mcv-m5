###### Codigo para lanzar dos experimentos seguidos.

import subprocess
import os

# Plantilla para comandos:
cmd_mask = "python train.py -c %s -e %s"

# Experimento 1:
config_file_1 = '/config/tt100k_inception_debug.py'
experiment_name_1 = 'prueba_lanza_1'
cmd1 = cmd_mask % (config_file_1, experiment_name_1)

# Experimento 2:
config_file_2 = '/config/tt100k_inception_debug.py'
experiment_name_2 = 'prueba_lanza_1'
cmd2 = cmd_mask % (config_file_2, experiment_name_2)

# Ejecuciones:
os.system('export CUDA_VISIBLE_DEVICES=0')
subprocess.call(cmd1)
subprocess.call(cmd2)