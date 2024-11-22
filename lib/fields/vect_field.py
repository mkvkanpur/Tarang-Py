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
from lib.global_fns.universal import Universal_arrays



if para.device == "gpu":
      import cupy as ncp

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp

class VectorField:

    def __init__(self):
        self.Vkx = []
        self.Vky = []
        self.Vkz = []
        self.Vx = []
        self.Vy = []
        self.Vz = []
        self.force_Vx = []
        self.force_Vy = []
        self.force_Vz = []
        self.nlinx = []
        self.nliny = []
        self.nlinz = []
        self.ek = []
        self.Tk = []
        self.ek_table = []
        self.Tk_table = []
        self.total_energy = []
        self.standard_dissipation = []
       
        self.ur = []
        self.Vkx_modes_t = []
        self.Vky_modes_t = []
        self.Vkz_modes_t = []
        self.Vkx_mode = []  # list of list [Vkx[mode], .... ...]
        self.Vkz_mode = []  # list of list [Vkz[mode], .... ...]
        self.Vky_mode = []


    def set_arrays(self):

        if (para.dimension == 2):
            self.Vkx = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
            self.Vkz = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
            self.Vx = ncp.zeros([para.Nx, para.Nz], dtype=para.real_dtype)
            self.Vz = ncp.zeros([para.Nx, para.Nz], dtype=para.real_dtype)
            self.force_Vx = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
            self.force_Vz = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
            self.nlinx = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
            self.nlinz = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)

        if (para.dimension == 3):
            self.Vkx = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
            self.Vky = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
            self.Vkz = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
            self.Vx = ncp.zeros([para.Nx, para.Ny, para.Nz], dtype=para.real_dtype)
            self.Vy = ncp.zeros([para.Nx, para.Ny, para.Nz], dtype=para.real_dtype)
            self.Vz = ncp.zeros([para.Nx, para.Ny, para.Nz], dtype=para.real_dtype)
            self.force_Vx = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
            self.force_Vy = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
            self.force_Vz = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
            self.nlinx = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
            self.nliny = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
            self.nlinz = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)

        self.ek = ncp.zeros((min_radius_outside+1, para.dimension), dtype=para.real_dtype)
        self.Tk = ncp.zeros((min_radius_outside+1, para.dimension), dtype=para.real_dtype)
        # SHASHWAT: min_radius_outside is now kept independant of box size and only depends on Ngrid
        # Changed ek Tk structure to match with CUDA code

        self.ur = ncp.zeros(3, dtype=para.real_dtype)

        return


    def init_cond(self,  Vkx, Vkz, Vky=[]):
        self.Vkz[...] = Vkz[...]

        if (para.dimension > 1):
            self.Vkx[...] = Vkx[...]

        if (para.dimension > 2):
            self.Vky[...] = Vky[...]

        return



    def get_divergence(self):
        if (para.dimension == 2):
            return ncp.max(ncp.abs(xderiv(self.Vkx) + zderiv(self.Vkz)))

        if (para.dimension == 3):
            return ncp.max(ncp.abs(xderiv(self.Vkx) + yderiv(self.Vky) + zderiv(self.Vkz)))

        return


    def compute_total_energy(self):
        if (para.dimension == 2):
            totE = ncp.sum(ncp.abs(self.Vkx)**2 + ncp.abs(self.Vkz)**2)
            totE -= ncp.sum(ncp.abs(self.Vkx[...,0])**2+ncp.abs(self.Vkz[...,0])**2)/2

        elif (para.dimension == 3):
            totE = ncp.sum(ncp.abs(self.Vkx)**2 + ncp.abs(self.Vky)**2 + ncp.abs(self.Vkz)**2)
            totE -= ncp.sum(ncp.abs(self.Vkx[...,0])**2+ncp.abs(self.Vky[...,0])**2+ncp.abs(self.Vkz[...,0])**2)/2

        return totE


    def compute_dissipation(self):
        if (para.dimension == 2):
            dissip = 2*ncp.sum(ksqr*(ncp.abs(self.Vkx)**2 + ncp.abs(self.Vkz)**2))
            dissip -= 2*ncp.sum(ksqr[...,0]*(ncp.abs(self.Vkx[...,0])**2+ncp.abs(self.Vkz[...,0])**2))/2

        elif (para.dimension == 3):
            dissip = 2*ncp.sum(ksqr*(ncp.abs(self.Vkx)**2 + ncp.abs(self.Vky)**2 + ncp.abs(self.Vkz)**2))
            dissip -= 2*ncp.sum(ksqr[...,0]*(ncp.abs(self.Vkx[...,0])**2+ncp.abs(self.Vky[...,0])**2+ncp.abs(self.Vkz[...,0])**2))/2

        return dissip


    def compute_energy_spectrum_Tk(self, univ = Universal_arrays()):

        if (para.dimension == 1):
            self.ek[...,-1] = ncp.abs(self.Vkz[:])**2
            self.ek[0,  -1] /= 2
            self.Tk[...,-1] = -(self.nlinz[:]*ncp.conjugate(self.Vkz[:])).real

        elif (para.dimension == 2):
            univ.temp_nlin[...] = self.Vkx[...]
            univ.temp_nlin[-1:para.Nx//2:-1,0] = complex(0, 0)

            self.ek[0, 0] = abs(self.Vkx[0,0])**2/2
            self.Tk[0, 0] = -(self.nlinx[0,0]*ncp.conjugate(self.Vkx[0,0])).real

            # Tk_p = ncp.sum((self.nlinx[...]*ncp.conjugate(univ.temp_nlin[...])).real)

            for k in range(1, min_radius_outside+1):
                index = ncp.where((ksqr > (k_array[k-1])**2) & (ksqr <= (k_array[k])**2))
                self.ek[k, 0] = ncp.sum(ncp.abs(univ.temp_nlin[index])**2)
                self.Tk[k, 0] = -2.0*ncp.sum(((self.nlinx[...]*ncp.conjugate(univ.temp_nlin[...])).real)[index])

            univ.temp_nlin[...] = self.Vkz[...]
            univ.temp_nlin[-1:para.Nx//2:-1,0] = complex(0, 0)

            self.ek[0, -1] = abs(self.Vkz[0,0])**2/2
            self.Tk[0, -1] += -(self.nlinz[0,0]*ncp.conjugate(self.Vkz[0,0])).real

            # Tk_p += ncp.sum((self.nlinz[...]*ncp.conjugate(univ.temp_nlin[...])).real)

            for k in range(1, min_radius_outside+1):
                index = ncp.where((ksqr > (k_array[k-1])**2) & (ksqr <= (k_array[k])**2))
                self.ek[k, -1] = ncp.sum(ncp.abs(univ.temp_nlin[index])**2)
                self.Tk[k, -1] += -2.0*ncp.sum(((self.nlinz[...]*ncp.conjugate(univ.temp_nlin[...])).real)[index])

        elif (para.dimension == 3):
            univ.temp_nlin[...] = self.Vkx[...]
            univ.temp_nlin[-1:para.Nx//2:-1,:, 0] = complex(0, 0)
            univ.temp_nlin[0,para.Ny-1:para.Ny//2:-1,0] = complex(0, 0)

            self.ek[0, 0] = abs(self.Vkx[0,0,0])**2/2
            self.Tk[0, 0] = -(self.nlinx[0,0,0]*ncp.conjugate(self.Vkx[0,0,0])).real

            for k in range(1, min_radius_outside+1):
                index = ncp.where((ksqr > (k_array[k-1])**2) & (ksqr <= (k_array[k])**2))
                self.ek[k, 0] = ncp.sum(ncp.abs(univ.temp_nlin[index])**2)
                self.Tk[k, 0] = -2.0*ncp.sum(((self.nlinx[...]*ncp.conjugate(univ.temp_nlin[...])).real)[index])

            univ.temp_nlin[...] = self.Vky[...]
            univ.temp_nlin[-1:para.Nx//2:-1,:,0] = complex(0, 0)
            univ.temp_nlin[0,para.Ny-1:para.Ny//2:-1,0] = complex(0, 0)

            self.ek[0, 1] = abs(self.Vky[0,0,0])**2/2
            self.Tk[0, 1] = -(self.nliny[0,0,0]*ncp.conjugate(self.Vky[0,0,0])).real

            for k in range(1, min_radius_outside+1):
                index = ncp.where((ksqr > (k_array[k-1])**2) & (ksqr <= (k_array[k])**2))
                self.ek[k, 1] = ncp.sum(ncp.abs(univ.temp_nlin[index])**2)
                self.Tk[k, 1] = -2.0*ncp.sum(((self.nliny[...]*ncp.conjugate(univ.temp_nlin[...])).real)[index])

            univ.temp_nlin[...] = self.Vkz[...]
            univ.temp_nlin[-1:para.Nx//2:-1,:,0] = complex(0, 0)
            univ.temp_nlin[0,para.Ny-1:para.Ny//2:-1,0] = complex(0, 0)

            self.ek[0, -1] = abs(self.Vkz[0,0,0])**2/2
            self.Tk[0, -1] = -(self.nlinz[0,0,0]*ncp.conjugate(self.Vkz[0,0,0])).real

            for k in range(1, min_radius_outside+1):
                index = ncp.where((ksqr > (k_array[k-1])**2) & (ksqr <= (k_array[k])**2))
                self.ek[k, -1] = ncp.sum(ncp.abs(univ.temp_nlin[index])**2)
                self.Tk[k, -1] = -2.0*ncp.sum(((self.nlinz[...]*ncp.conjugate(univ.temp_nlin[...])).real)[index])

        return




    def U_to_Ucopy(self, univ = Universal_arrays()):
        if (para.dimension == 2):
            univ.temp1_RK[...] = self.Vkx[...]
            univ.temp2_RK[...] = self.Vkz[...]

        elif (para.dimension == 3):
            univ.temp1_RK[...] = self.Vkx[...]
            univ.temp2_RK[...] = self.Vkz[...]
            univ.temp3_RK[...] = self.Vky[...]

        return


    def Ucopy_to_U(self, univ= Universal_arrays()):
        if (para.dimension == 2):
            self.Vkx[...] = univ.temp1_RK[...]
            self.Vkz[...] = univ.temp2_RK[...]

        elif (para.dimension == 3):
            self.Vkx[...] = univ.temp1_RK[...]
            self.Vkz[...] = univ.temp2_RK[...]
            self.Vky[...] = univ.temp3_RK[...]

        return


