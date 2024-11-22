# Copyright (c) 2024 Mahendra Verma. All rights reserved.

# Authors:
# - Mahendra Verma
# - Soumyadeep Chatterjee
# - Shashwat Nirgudkar

# This software is provided for non-commercial use only. You may not use, copy, modify, distribute, or otherwise exploit the software or any derivative works of the software for commercial purposes. Specifically, you may not sell, license, or profit from the software in any manner.

# Usage of the software is strictly limited to personal, academic, or non-commercial purposes. Any use, modification, or redistribution of this software outside of these constraints is prohibited without the explicit written consent of the copyright holder.

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




import para as para
import random
import lib.global_fns.spectral_setup as spectral_setup
from lib.fields.vect_field import VectorField
from lib.global_fns.universal import Universal_arrays

if para.device == "gpu":
      import cupy as ncp

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp


def compute_force_hydro(U = VectorField(), univ = Universal_arrays(), force_Vx=[], force_Vz=[], force_Vy=[]):

    if not para.forcing_enabled:
        U.force_Vz = U.force_Vz*0
        if (para.dimension > 1):
            U.force_Vx = U.force_Vx*0
        if (para.dimension > 2):
            U.force_Vy = U.force_Vy*0

    elif (para.dimension == 1):
        U.force_Vz = U.force_Vz*0

    elif (para.dimension == 2):
        if para.device == "gpu":
            ncp._core.set_routine_accelerators(['cub', 'cutensor'])

        index_forcing_shell = ncp.where((spectral_setup.ksqr > (para.forcing_range[0])**2) & (spectral_setup.ksqr <= (para.forcing_range[1])**2))

        if (univ.force_phase):
            univ.phase_ch1 = 2*ncp.pi * random.random();
            univ.force_phase = False

        energy_supply_k_zp = para.injection_rate/(index_forcing_shell[-1].size)
        # SHASHWAT: Change to forcing range instead of single index
        # Shifted functions to spectral_setup and removed dependancy on No_of_modes
        # Code now directly gets the modes through index_forcing_shell

        for i in range(ncp.shape(index_forcing_shell[0])[0]):

            kx = int(spectral_setup.kx_mesh[index_forcing_shell[0][i], index_forcing_shell[1][i]])
            kz = int(spectral_setup.kz_mesh[index_forcing_shell[0][i], index_forcing_shell[1][i]])

            Kmag = ncp.sqrt(spectral_setup.ksqr[index_forcing_shell[0][i], index_forcing_shell[1][i]])

            angle = spectral_setup.compute_phi(kx, kz)

            U1 = (U.Vkx[kx, kz]*ncp.sin(angle) - U.Vkz[kx, kz]*ncp.cos(angle));

            P_artificial = univ.dt/(2.0*ncp.pi*Kmag);
            P_physical =  ncp.real(ncp.conjugate(U1)*(1.0/ncp.sqrt(ncp.pi*Kmag))*ncp.exp(1j*univ.phase_ch1))

            if (P_physical >= 0):
                c = (-P_physical+ncp.sqrt(P_physical*P_physical+4.0*P_artificial*energy_supply_k_zp))/(2.0*P_artificial);
            elif(P_physical < 0):
                c = (-P_physical-ncp.sqrt(P_physical*P_physical+4.0*P_artificial*energy_supply_k_zp))/(2.0*P_artificial);

            # print(kx, kz, "P_artificial", (c**2*P_artificial), "P_physical", P_physical*c, "total", (c**2*P_artificial)+P_physical*c)
            f1 = (c/ncp.sqrt(ncp.pi*Kmag))*ncp.exp(1j*univ.phase_ch1);

            U.force_Vx[kx,kz], U.force_Vz[kx,kz] = spectral_setup.craya_to_cartesian(f1, U.force_Vx[kx,kz], U.force_Vz[kx,kz], Kmag, kx, kz)

        U.force_Vx,  U.force_Vz = spectral_setup.reality_cond(U.force_Vx), spectral_setup.reality_cond(U.force_Vz)

        supply = 2*ncp.sum(ncp.real((U.force_Vx*ncp.conjugate(U.Vkx)) + (U.force_Vz*ncp.conjugate(U.Vkz))))
        supply -= ncp.sum(ncp.real((U.force_Vx[:,0]*ncp.conjugate(U.Vkx[:,0])) + (U.force_Vz[:,0]*ncp.conjugate(U.Vkz[:,0]))))

        # print("supply", supply)


    elif (para.dimension == 3):
        U.force_Vx[1,1,1] =   0-0.125j     # u = Sin(x)Cos(y)Cos(z)
        U.force_Vx[1,-1,1] =   0-0.125j     # u = Sin(x)Cos(y)Cos(z)
        U.force_Vx[-1,1,1] =   0+0.125j     # u = Sin(x)Cos(y)Cos(z)
        U.force_Vx[-1,-1,1] =   0+0.125j     # u = Sin(x)Cos(y)Cos(z)

        U.force_Vy[1,1,1] =   0+0.125j     # u = Sin(x)Cos(y)Cos(z)
        U.force_Vy[1,-1,1] =   0-0.125j     # u = Sin(x)Cos(y)Cos(z)
        U.force_Vy[-1,1,1] =   0+0.125j     # u = Sin(x)Cos(y)Cos(z)
        U.force_Vy[-1,-1,1] =   0-0.125j     # u = Sin(x)Cos(y)Cos(z)

    return

