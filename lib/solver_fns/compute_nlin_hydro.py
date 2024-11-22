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
from lib.io.io_hydro import *

if para.device == "gpu":
      import cupy as ncp

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp


def compute_nlin_u(t, U = VectorField(), univ=Universal_arrays()):
    if (para.dimension == 2):

        univ.temp_nlin[:, :] = U.Vkx[:, :]
        U.Vkx = dealias(U.Vkx)
        U.Vx = inverse_transform(U.Vkx, U.Vx)
        U.Vkx[:, :] =  univ.temp_nlin[:, :]

        univ.temp_nlin[:, :] = U.Vkz[:, :]
        U.Vkz = dealias(U.Vkz)
        U.Vz = inverse_transform(U.Vkz, U.Vz)
        U.Vkz[:, :] =  univ.temp_nlin[:, :]

        if (not para.FIXED_DT) and (univ.compute_dt):
            Get_dt_hydro(U, univ)
            univ.compute_dt = False

        univ.tempR[:, :] = (U.Vx[:, :])**2
        U.nlinx = forward_transform(univ.tempR, U.nlinx)
        U.nlinx = xderiv(U.nlinx)

        univ.tempR[:, :] = (U.Vz[:, :])**2
        U.nlinz = forward_transform(univ.tempR, U.nlinz)
        U.nlinz = zderiv(U.nlinz)

        # off-diagnoal terms
        univ.tempR[:, :] = (U.Vx[:, :])*(U.Vz[:, :])
        univ.temp_nlin = forward_transform(univ.tempR, univ.temp_nlin)
        U.nlinx += zderiv(univ.temp_nlin)
        U.nlinz += xderiv(univ.temp_nlin)


    elif (para.dimension == 3):

        univ.temp_nlin[:, :,:] = U.Vkx[:, :,:]
        U.Vkx = dealias(U.Vkx)
        U.Vx = inverse_transform(U.Vkx, U.Vx)
        U.Vkx[:, :,:] =  univ.temp_nlin[:, :,:]

        univ.temp_nlin[:, :,:] = U.Vky[:, :,:]
        U.Vky = dealias(U.Vky)
        U.Vy = inverse_transform(U.Vky, U.Vy)
        U.Vky[:, :,:] =  univ.temp_nlin[:, :,:]

        univ.temp_nlin[:, :,:] = U.Vkz[:, :,:]
        U.Vkz = dealias(U.Vkz)
        U.Vz = inverse_transform(U.Vkz, U.Vz)
        U.Vkz[:, :,:] =  univ.temp_nlin[:, :,:]

        if (not para.FIXED_DT) and (univ.compute_dt):
            Get_dt_hydro(U, univ)
            univ.compute_dt = False

        univ.tempR[:, :, :] = (U.Vx[:, :, :])**2
        U.nlinx = forward_transform(univ.tempR, U.nlinx)
        U.nlinx = xderiv(U.nlinx)

        univ.tempR[:, :, :] = (U.Vy[:, :, :])**2
        U.nliny = forward_transform(univ.tempR, U.nliny)
        U.nliny = yderiv(U.nliny)

        univ.tempR[:, :, :] = (U.Vz[:, :, :])**2
        U.nlinz = forward_transform(univ.tempR, U.nlinz)
        U.nlinz = zderiv(U.nlinz)

        # off-diagonal terms
        univ.tempR[:, :, :] = U.Vx[:, :, :]
        U.Vx *= U.Vy
        U.Vy *= U.Vz
        U.Vz *= univ.tempR

        univ.temp_nlin = forward_transform(U.Vx, univ.temp_nlin)
        U.nlinx += yderiv(univ.temp_nlin)
        U.nliny += xderiv(univ.temp_nlin)

        univ.temp_nlin = forward_transform(U.Vy, univ.temp_nlin)
        U.nliny += zderiv(univ.temp_nlin)
        U.nlinz += yderiv(univ.temp_nlin)

        univ.temp_nlin = forward_transform(U.Vz, univ.temp_nlin)
        U.nlinx += zderiv(univ.temp_nlin)
        U.nlinz += xderiv(univ.temp_nlin)

    if (univ.compute_nlin_first_flag):
      output_ekTk_hydro(t,  U, univ)
      univ.compute_nlin_first_flag = False

    return

def Get_dt_hydro(U = VectorField(), univ= Universal_arrays()):

    U.ur[0] = ncp.max(ncp.abs(U.Vx))
    U.ur[-1] = ncp.max(ncp.abs(U.Vz))

    univ.ubydx = U.ur[0]/univ.dx + U.ur[-1]/univ.dz

    if (para.dimension == 3):
        U.ur[1] = ncp.max(ncp.abs(U.Vy))

        univ.ubydx += U.ur[1]/univ.dy

    if (univ.ubydx == 0):
        univ.dt = univ.dt_fixed

        return

    univ.dt = para.Courant_no/univ.ubydx

    univ.dt = min(univ.dt, univ.dt_fixed)

    return

