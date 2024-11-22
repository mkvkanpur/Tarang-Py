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

if para.device == "gpu":
      import cupy as ncp

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp


class Universal_arrays:

    def __init__(self):
        self.compute_nlin_first_flag = None
        self.compute_dt = None
        self.dt_fixed = 0
        self.dt = 0
        self.t_last = False
        self.force_phase = None
        self.phase_ch1 = 0
        self.iter = 0
        self.ubydx = 0
        self.exp_factor_array = []
        self.temp_nlin = [] # temp in Fourier space
        self.temp1_RK = [] # temp in Fourier space
        self.temp2_RK = []
        self.temp3_RK = []
        self.temp1b_RK = []
        self.temp2b_RK = []
        self.temp3b_RK = []
        self.temp_scalar1_RK = []
        self.temp_scalar2_RK = []
        self.tempR = []  # temp array in real space
        self.dx = None
        self.dy = None
        self.dz = None
        self.t = []
        self.t_field_save = []
        self.t_glob_energy_print = []
        self.t_ekTk_save = []
        self.t_modes_save = []
        self.cross_helicity = []


    def set_arrays(self):

        

        if (para.dimension == 2):
            if (para.kind == "HYDRO") or (para.kind == "HYDRO3"):
                self.exp_factor_array = ncp.zeros([para.Nx,  para.Nz//2+1], dtype=para.real_dtype)
           
            self.tempR = ncp.zeros([para.Nx, para.Nz], dtype = para.real_dtype)
            self.temp_nlin = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
            
            if (para.time_scheme == "RK2"):
                self.temp1_RK = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp2_RK = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)

            

            if (para.time_scheme == "RK4"):
                self.temp1_RK = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp2_RK = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp1b_RK = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp2b_RK = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)

          
        if (para.dimension == 3):
            
            if (para.kind == "HYDRO"):
                self.exp_factor_array = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.real_dtype)
            

            self.tempR = ncp.zeros([para.Nx, para.Ny, para.Nz], dtype = para.real_dtype)
            self.temp_nlin = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)

            if (para.time_scheme == "RK2"):
                self.temp1_RK = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp2_RK = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp3_RK = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)

            if (para.time_scheme == "RK4"):
                self.temp1_RK = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp2_RK = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp3_RK = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp1b_RK = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp2b_RK = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)
                self.temp3b_RK = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)

        return


    def set_exp_arrays(self):

        if (para.FIXED_DT):
            if (para.kind == "HYDRO"):   
                self.exp_factor_array = ncp.exp(-(para.nu*(ksqr))*para.dt)

        else:
            if (para.kind == "HYDRO"):
                    self.exp_factor_array = ncp.exp(-(para.nu*(ksqr)))

        return


    def set_grid_space(self):

        if para.box_size_default:
            if para.dimension == 1:
                self.dx = 2*ncp.pi/para.Nx

            elif para.dimension == 2:
                self.dx, self.dz  = 2*ncp.pi/para.Nx, 2*ncp.pi/para.Nz

            elif para.dimension == 3:
                self.dx, self.dy, self.dz  = 2*ncp.pi/para.Nx, 2*ncp.pi/para.Ny, 2*ncp.pi/para.Nz

        else:
            if para.dimension == 1:
                self.dx  = para.L[0]/para.Nx

            elif para.dimension == 2:
                self.dx, self.dz  = para.L[0]/para.Nx, para.L[2]/para.Nz

            elif para.dimension == 3:
                self.dx, self.dy, self.dz  = para.L[0]/para.Nx, para.L[1]/para.Ny, para.L[2]/para.Nz

        return



