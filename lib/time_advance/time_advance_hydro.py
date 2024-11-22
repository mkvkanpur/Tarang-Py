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
from lib.solver_fns.compute_nlin_hydro import *
from lib.fields.pressure import Pressure
from lib.global_fns.universal import Universal_arrays
from lib.io.io_hydro import *

from lib.force.compute_force_hydro import *

if para.device == "gpu":
    import cupy as ncp

    dev1 = ncp.cuda.Device(para.device_rank)

    dev1.use()

else:
    import numpy as ncp


# Yields N(U) = -N(U) -grad(p) + F
def compute_rhs_hydro(t, U=VectorField(), P=Pressure(), univ=Universal_arrays()):
    compute_force_hydro(U, univ)

    compute_nlin_u(t, U, univ)

    P.compute_pressure_u(U)

    U.nlinx +=  -U.force_Vx
    U.nlinz +=  -U.force_Vz
    if (para.dimension == 3):
        U.nliny +=  -U.force_Vy

    # Add pressure grad to RHS. nlin is the RHS
    U.nlinx +=  xderiv(P.p)
    U.nlinz +=  zderiv(P.p)

    if (para.dimension == 3):
        U.nliny +=  yderiv(P.p)

    U.nlinx *= -1
    U.nlinz *= -1

    if (para.dimension == 3):
        U.nliny *= -1

    return


def time_adv_single_step_hydro(a, b, c, U=VectorField(), univ=Universal_arrays()):
    if (para.FIXED_DT):
        if (a == 1 and b == 1):
            U.Vkx +=  c*univ.dt_fixed * U.nlinx
            U.Vkz +=  c*univ.dt_fixed * U.nlinz

            U.Vkx *= univ.exp_factor_array
            U.Vkz *= univ.exp_factor_array

            if (para.dimension == 3):
                U.Vky +=  c*univ.dt_fixed * U.nliny
                U.Vky *= univ.exp_factor_array

            return

        elif (a == 1/2 and b == 0):
            U.Vkx *= ncp.sqrt(univ.exp_factor_array)
            U.Vkz *= ncp.sqrt(univ.exp_factor_array)

            U.Vkx +=  c*univ.dt_fixed * U.nlinx
            U.Vkz +=  c*univ.dt_fixed * U.nlinz


            if (para.dimension == 3):
                U.Vky *= ncp.sqrt(univ.exp_factor_array)
                U.Vky +=  c*univ.dt_fixed * U.nliny

            return

        elif (a == 1 and b == 1/2):
            U.Vkx *= ncp.sqrt(univ.exp_factor_array)
            U.Vkz *= ncp.sqrt(univ.exp_factor_array)

            U.Vkx +=  c*univ.dt_fixed * U.nlinx
            U.Vkz +=  c*univ.dt_fixed * U.nlinz

            U.Vkx *= ncp.sqrt(univ.exp_factor_array)
            U.Vkz *= ncp.sqrt(univ.exp_factor_array)

            if (para.dimension == 3):
                U.Vky *= ncp.sqrt(univ.exp_factor_array)
                U.Vky +=  c*univ.dt_fixed * U.nliny
                U.Vky *= ncp.sqrt(univ.exp_factor_array)

            return

        elif (a == 1/2 and b == 1/2):
            U.Vkx +=  c*univ.dt_fixed * U.nlinx
            U.Vkz +=  c*univ.dt_fixed * U.nlinz

            U.Vkx *= ncp.sqrt(univ.exp_factor_array)
            U.Vkz *= ncp.sqrt(univ.exp_factor_array)

            if (para.dimension == 3):
                U.Vky +=  c*univ.dt_fixed * U.nliny
                U.Vky *= ncp.sqrt(univ.exp_factor_array)

            return
    else:
        if (para.time_scheme == "EULER"):
            U.Vkx +=  univ.dt * U.nlinx
            U.Vkz +=  univ.dt * U.nlinz

            U.Vkx *= univ.exp_factor_array**univ.dt
            U.Vkz *= univ.exp_factor_array**univ.dt

            if (para.dimension == 3):
                U.Vky +=  univ.dt * U.nliny
                U.Vky *= univ.exp_factor_array**univ.dt

            return

        else:

            U.Vkx *= univ.exp_factor_array**((a-b) * univ.dt)
            U.Vkz *= univ.exp_factor_array**((a-b) * univ.dt)

            U.Vkx +=  c*univ.dt * U.nlinx
            U.Vkz +=  c*univ.dt * U.nlinz

            U.Vkx *= univ.exp_factor_array**((b) * univ.dt)
            U.Vkz *= univ.exp_factor_array**((b) * univ.dt)

            if (para.dimension == 3):
                U.Vky *= univ.exp_factor_array**((a-b) * univ.dt)
                U.Vky += c*univ.dt * U.nliny
                U.Vky *= univ.exp_factor_array**((b) * univ.dt)

            return


def time_advance_Euler_hydro(U=VectorField(),P=Pressure(), univ=Universal_arrays()):
    t = para.tinit

    univ.dt_fixed = para.dt

    if (not para.FIXED_DT):
        univ.dt = univ.dt_fixed

    while (t < para.tfinal):
        if (ncp.isnan(U.compute_total_energy())) or ((not para.FIXED_DT) and (univ.dt < 1E-7)):
            univ.T_last = True
            break

        univ.compute_nlin_first_flag = True

        univ.compute_dt = True

        output_hydro(t, U, univ)

        #print(t, univ.dt, U.compute_total_energy(), U.get_divergence())

        compute_rhs_hydro(t, U, P, univ)

        univ.iter += 1
        if (para.FIXED_DT):
            t = para.tinit + univ.iter*univ.dt_fixed
        else:
            t += univ.dt

        if para.device == "gpu":
            univ.t.append(ncp.asnumpy(t))
        else:
            univ.t.append(t)

        time_adv_single_step_hydro(1, 1, 1, U, univ)

        #print(t, univ.dt_fixed, U.compute_total_energy(), U.get_divergence())

        U.Vkx, U.Vkz = reality_cond(U.Vkx), reality_cond(U.Vkz)

        if (para.dimension == 3):
            U.Vky = reality_cond(U.Vky)

    # For CFL, when t > tfinal, print the last field
    univ.t_last = True
    output_hydro(univ.t[-1], U, univ)
    output_ekTk_hydro(univ.t[-1], U, univ)

    return


def time_advance_RK2_hydro(U=VectorField(), P=Pressure(), univ=Universal_arrays()):
    t = para.tinit

    univ.dt_fixed = para.dt

    if (not para.FIXED_DT):
        univ.dt = univ.dt_fixed

    while (t < para.tfinal):
        if (ncp.isnan(U.compute_total_energy())) or ((not para.FIXED_DT) and (univ.dt < 1E-7)):
            univ.T_last = True
            break

        # Now Ucopy = U
        U.U_to_Ucopy(univ)

        univ.compute_nlin_first_flag = True

        univ.compute_dt = True

        output_hydro(t, U, univ)

        compute_rhs_hydro(t, U, P, univ)

        univ.iter += 1;
        if (para.FIXED_DT):
            t = para.tinit + univ.iter*univ.dt_fixed
        else:
            t += univ.dt

        if para.device == "gpu":
            univ.t.append(ncp.asnumpy(t))
        else:
            univ.t.append(t)

        # Go to mid point
        time_adv_single_step_hydro(1/2, 1/2, 1/2, U, univ)

        compute_rhs_hydro(t, U, P, univ)

        # U.nlin contains rhs with Umid

        U.Ucopy_to_U(univ)

        time_adv_single_step_hydro(1, 1/2, 1, U, univ)

        U.Vkx, U.Vkz = reality_cond(U.Vkx), reality_cond(U.Vkz)

        if (para.dimension == 3):
            U.Vky = reality_cond(U.Vky)

    # For CFL, when t > tfinal, print the last field
    univ.t_last = True
    output_hydro(univ.t[-1], U, univ)
    output_ekTk_hydro(univ.t[-1],  U, univ)
    return
