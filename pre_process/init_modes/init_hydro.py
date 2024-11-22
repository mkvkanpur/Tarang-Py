# Copyright (c) 2024 Mahendra Verma. All rights reserved.

# Authors:
# - Mahendra Verma
# - Soumyadeep Chatterjee
# - Shashwat Nirgudkar

# This software is provided for non-commercial use only. You may not use, copy, modify, distribute, or otherwise exploit the software or any derivative works of the software for commercial purposes. Specifically, you may not sell, license, or profit from the software in any manner.

# Usage of the software is strictly limited to personal, academic, or non-commercial purposes. Any use, modification, or redistribution of this software outside of these constraints is prohibited without the explicit written consent of the copyright holder.

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



import sys

sys.path.insert(0, '../../tarang')

import para as para
import lib.global_fns.spectral_setup as spectral_setup
import h5py

if para.device == "gpu":
      import cupy as ncp

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp

from lib.global_fns.spectral_setup import *

##2d run ----------------------
#set init cond array

def init_hydro_main():
    if (para.dimension == 2):
        Vkx = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=complex)
        Vkz = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=complex)
        Vky = []

        #Vkx[1,0] = 0+0j
        #Vkx[0,1] = 1+0j
        #Vkx[1,1] = 0+1j
        #Vkx[-1,1] = 0-1j

        index = ncp.where((spectral_setup.ksqr > (10)**2) & (spectral_setup.ksqr <= (80)**2) & (spectral_setup.kz_mesh != 0))
        index2 = ncp.where((spectral_setup.ksqr > (10)**2) & (spectral_setup.ksqr <= (80)**2) & (spectral_setup.kz_mesh == 0))
        # print(index)
        # print(index2)

        Vkx[index] = 0.001*(ncp.random.rand() + 1j*ncp.random.rand())

        Vkz[index] = -(spectral_setup.kx_mesh[index]*Vkx[index])/spectral_setup.kz_mesh[index]#0.01*(ncp.random.rand() + 1j*ncp.random.rand())

        Vkx[index2] = complex(0,0)
        Vkz[index2] = complex(0,0)
        '''
        Vkx[3,0] = 0+0j
        Vkx[0,3] = 1+0j
        Vkx[3,3] = 0+1j
        Vkx[-3,3] = 0-1j

        Vkx[6,0] = 0+0j
        Vkx[0,6] = 1+0j
        Vkx[6,6] = 0+1j
        Vkx[-6,6] = 0-1j
        #Vkx[-1,0] = 0+0j

        Vkz[1,0] = 1+0j
        Vkz[0,1] = 0+0j
        Vkz[1,1] = 0-1j
        Vkx[-1,1] = 0-1j

        Vkz[3,0] = 1+0j
        Vkz[0,3] = 0+0j
        Vkz[3,3] = 0-1j
        Vkx[-3,3] = 0-1j

        Vkz[6,0] = 1+0j
        Vkz[0,6] = 0+0j
        Vkz[6,6] = 0-1j
        Vkx[-6,6] = 0-1j
        #Vkz[-1,0] = 1+0j
        '''
        Vkx, Vkz = reality_cond(Vkx), reality_cond(Vkz)

#--------------------------------------------


#--------------------------------------------

#3d run ----------------------
#set init cond array


    if (para.dimension == 3):
        Vkx = ncp.zeros((para.Nx,para.Ny, (para.Nz)//2+1),dtype= complex)
        Vky = ncp.zeros((para.Nx,para.Ny, (para.Nz)//2+1),dtype= complex)
        Vkz = ncp.zeros((para.Nx,para.Ny, (para.Nz)//2+1),dtype= complex)


        Vkx[1,1,2] = 0+1j
        Vkx[1,0,1] = 0-1j
        Vkx[0,1,1] = 0+0j
        #Vkx[1,-1,2] = 0+1j
        #Vkx[-1,1,2] = 0-1j
        #Vkx[-1,0,1] = 0+1j
        #Vkx[0,-1,1] = 0+0j

        Vky[1,1,2] = 0+1j
        Vky[1,0,1] = 0+0j
        Vky[0,1,1] = 0-1j
        #Vky[1,-1,2] = 0-1j
        #Vky[-1,1,2] = 0+1j
        #Vky[-1,0,1] = 0+0j
        #Vky[0,-1,1] = 0+1j

        Vkz[1,1,2] = 0-1j
        Vkz[1,0,1] = 0+1j
        Vkz[0,1,1] = 0+1j
        #Vkz[1,-1,2] = 0-1j
        #Vkz[-1,1,2] = 0-1j
        #Vkz[-1,0,1] = 0+1j
        #Vkz[0,-1,1] = 0+1j

        Vkx, Vky, Vkz = reality_cond(Vkx), reality_cond(Vky), reality_cond(Vkz)


    hf = h5py.File(para.input_dir+"/"+para.input_file_name, 'w')
    if para.device == "gpu":
        hf.create_dataset('Vkx', data=ncp.asnumpy(Vkx))
        hf.create_dataset('Vkz', data=ncp.asnumpy(Vkz))
        if (para.dimension == 3):
            hf.create_dataset('Vky', data=ncp.asnumpy(Vky))
    else:
        hf.create_dataset('Vkx', data=(Vkx))
        hf.create_dataset('Vkz', data=(Vkz))
        if (para.dimension == 3):
            hf.create_dataset('Vky', data=(Vky))

    hf.close()

if __name__ == "__main__":
    init_hydro_main()





