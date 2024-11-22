# Copyright (c) 2024 Mahendra Verma. All rights reserved.

# Authors:
# - Mahendra Verma
# - Soumyadeep Chatterjee
# - Shashwat Nirgudkar

# This software is provided for non-commercial use only. You may not use, copy, modify, distribute, or otherwise exploit the software or any derivative works of the software for commercial purposes. Specifically, you may not sell, license, or profit from the software in any manner.

# Usage of the software is strictly limited to personal, academic, or non-commercial purposes. Any use, modification, or redistribution of this software outside of these constraints is prohibited without the explicit written consent of the copyright holder.

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



import para as para
from lib.global_fns.spectral_setup import *
from lib.fields.vect_field import VectorField
from lib.global_fns.universal import Universal_arrays
from lib.fields.pressure import Pressure
from lib.time_advance.time_advance_hydro import *

from lib.io.io_hydro import *
import time
import shutil

if para.device == "gpu":
      import cupy as ncp

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp


def main_hydro():

    U = VectorField()

    P = Pressure()

    univ = Universal_arrays()

    U.set_arrays()
    P.set_arrays()
    univ.set_arrays()

    univ.set_exp_arrays()
    if not para.FIXED_DT:
        univ.set_grid_space()

    #assign initial condition
    try:
        initial_hydro(U)
    except:
        print("Invalid initial field")

    try:
        t_str_time_start = time.time()
        if para.time_scheme == "EULER":
            time_advance_Euler_hydro(U,P, univ)
        elif para.time_scheme == "RK2":
            time_advance_RK2_hydro(U,P, univ)
        elif para.time_scheme == "RK4":
            time_advance_RK4_hydro(U,P,univ)

    except (RuntimeWarning, UnboundLocalError) as err:
        print("Code blew up")

    finally:
        t_str_time_end = time.time()

        file_save_total_energy_hydro(U, univ)
        file_save_ekTk_hydro(U, univ)
        file_save_modes_hydro(U, univ)
        file_save_t_field_save_hydro(univ)
        file_save_t_hydro(univ)

        print("time loop compute time = ", t_str_time_end-t_str_time_start)

    #copy para in the output for future use
    shutil.copy2(para.para_directory + "/para.py", para.output_dir+"/para.py")

if __name__ == "__main__":
    main_hydro()
