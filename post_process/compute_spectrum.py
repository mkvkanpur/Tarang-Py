# Copyright (c) 2024 Mahendra Verma. All rights reserved.

# Authors:
# - Mahendra Verma
# - Soumyadeep Chatterjee
# - Shashwat Nirgudkar

# This software is provided for non-commercial use only. You may not use, copy, modify, distribute, or otherwise exploit the software or any derivative works of the software for commercial purposes. Specifically, you may not sell, license, or profit from the software in any manner.

# Usage of the software is strictly limited to personal, academic, or non-commercial purposes. Any use, modification, or redistribution of this software outside of these constraints is prohibited without the explicit written consent of the copyright holder.

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import para as para
from lib.fields.vect_field import VectorField
from lib.solver_fns.compute_nlin_mhd import *
from lib.global_fns.universal import Universal_arrays
from lib.io.io_mhd import *
import h5py

if para.device == "gpu":
      import cupy as ncp 

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp 

def main():
    if (para.dimension == 2):
        Vkx = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=complex)
        Vkz = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=complex)
        Vky = []


        Bkx = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=complex)
        Bkz = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=complex)
        Bky = []

    i = ncp.asarray([48.459773])#15.997545, 16.997545, 17.997545, 18.997545, 19.997545, 20.997545, 21.990736,  22.945394, 23.848059, 24.701595]) #,45.033611,48.265319])

    U = VectorField()

    W = VectorField()

    univ = Universal_arrays()

    U.set_arrays()
    W.set_arrays()
    univ.set_arrays(); j =0

    while j < 1:

        with h5py.File("Soln_%f.h5"%(i[j]),'r') as f:
            Vkx = ncp.asarray(f['Vkx'][()])
            Vkz = ncp.asarray(f['Vkz'][()])

            Bkx = ncp.asarray(f['Bkx'][()])
            Bkz = ncp.asarray(f['Bkz'][()]) 

        t = i[j]

        #U = VectorField()

        #W = VectorField()

        #univ = Universal_arrays()

        #U.set_arrays()
        #W.set_arrays()
        #univ.set_arrays()

        U.init_cond(Vkx, Vkz, Vky)
        W.init_cond(Bkx, Bkz, Bky)

        univ.compute_nlin_first_flag = True

        univ.compute_dt = False

        compute_nlin_mhd(t, U, W, univ)

    
        file_save_ekTk_mhd(U, W, univ)

        print(j)

        j += 1


if __name__ == "__main__":
    main()



