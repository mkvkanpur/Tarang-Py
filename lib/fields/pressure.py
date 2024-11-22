
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

if para.device == "gpu":
      import cupy as ncp

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp

class Pressure:

    def __init__(self):
        self.p = []


    def set_arrays(self):
        if (para.dimension == 2):
            self.p = ncp.zeros([para.Nx, para.Nz//2+1], dtype=para.complex_dtype)

        elif (para.dimension == 3):
            self.p = ncp.zeros([para.Nx, para.Ny, para.Nz//2+1], dtype=para.complex_dtype)


    def compute_pressure_u(self, U = VectorField()):
        if (para.dimension == 2):
            self.p[...] = (xderiv(U.nlinx[...]-U.force_Vx[...])
                           + zderiv(U.nlinz[...]-U.force_Vz[...]))

            ksqr[0,0] = 1
            self.p[...] = self.p[...]/ksqr[...]
            self.p[0,0] = complex(0, 0)
            ksqr[0,0] = 0

        elif (para.dimension == 3):
            self.p[...] = (xderiv(U.nlinx[...]-U.force_Vx[...])
                           + yderiv(U.nliny[...]-U.force_Vy[...])
                           + zderiv(U.nlinz[...]-U.force_Vz[...]))

            ksqr[0,0,0] = 1
            self.p[...] = self.p[...]/ksqr[...]
            self.p[0,0,0] = complex(0, 0)
            ksqr[0,0,0] = 0

        return self.p

