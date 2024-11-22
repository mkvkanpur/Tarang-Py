################################################################################

import numpy as np

########## Device details ######################################################

device = "CPU"              # "CPU" # "GPU"

device_rank = 0

########## Data-types ##########################################################

complex_dtype = "complex"
real_dtype = "float64"

########## Problem kind ########################################################

kind = "HYDRO"              # HYDRO # MHD # SCALAR # RBC # RBC_CHAOS # COARSEN

########## IO directories ######################################################

input_file_name = "init_cond.h5"
input_dir  = "input/"
output_dir = "output/"
para_directory = "."

########## Domain details ######################################################

dimension = 3               # 1 # 2 # 3

## Grid resolution
Nx = 64
Ny = 64                     # = 1 for 2d
Nz = 64

## Box size [Lx, Ly, Lz]
L = [10, 10, 10]

box_size_default = True     # Set True for a grid with 2pi edges

if (box_size_default):
    L = [2*np.pi, 2*np.pi, 2*np.pi]

kfactor = [2*np.pi/L[0], 2*np.pi/L[1], 2*np.pi/L[2]]

########## Dissipation constants ###############################################
# The form of dissipation is [nu * k^2 * U(k)]

nu = 2E-2                   # Viscosity # np.sqrt(Pr/Ra)

########## Forcing #############################################################


forcing_enabled = False

forcing_range = [4,6]       # forcing_range[0] < k <= forcing_range[1]

injection_rate = 0


########## Time advance ########################################################

time_scheme = "EULER"         # "EULER" "RK2" 


tinit = 0
tfinal = 1E-2
dt = 1E-3

FIXED_DT = True

Courant_no = 0.5

t_eps = 1e-8

########## Modes probe #########################################################


modes_save = [(1,0)]

########## Storage and Output ##################################################


# *_start define the time at which each parameter starts to be saved
# *_inter define the interval at which they get saved

iter_field_save_start = tinit
iter_field_save_inter = 100

iter_glob_energy_print_start = tinit
iter_glob_energy_print_inter = 1

iter_ekTk_save_start = tfinal
iter_ekTk_save_inter = 100

iter_modes_save_start = tinit
iter_modes_save_inter = 100

