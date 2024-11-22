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
import h5py

if para.device == "gpu":
      import cupy as ncp

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp

from lib.global_fns.spectral_setup import *

def main():
    if (para.dimension == 2):
        Vkx = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=complex)
        Vkz = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=complex)
        Vky = []

        Vx = ncp.zeros((para.Nx,para.Nz), dtype="float")
        Vz = ncp.zeros((para.Nx,para.Nz), dtype="float")
        Vy = []

    if (para.dimension == 3):
        Vkx = ncp.zeros((para.Nx,para.Ny, (para.Nz)//2+1),dtype= complex)
        Vky = ncp.zeros((para.Nx,para.Ny, (para.Nz)//2+1),dtype= complex)
        Vkz = ncp.zeros((para.Nx,para.Ny, (para.Nz)//2+1),dtype= complex)

        Vx = ncp.zeros((para.Nx,para.Ny, para.Nz),dtype= "float")
        Vy = ncp.zeros((para.Nx,para.Ny, para.Nz),dtype= "float")
        Vz = ncp.zeros((para.Nx,para.Ny, para.Nz),dtype= "float")

    with h5py.File(para.input_dir +"/Soln.h5",'r') as f:
        Vx = f['Vx'][()]
        Vz = f['Vz'][()]

        if para.dimension ==3:
            Vy = f['Vy'][()]

    Vkx, Vkz = forward_transform(Vx), forward_transform(Vz)

    Vkx, Vkz = reality_cond(Vkx), reality_cond(Vkz)

    if para.dimension ==3:
        Vky = forward_transform(Vy)

        Vky = reality_cond(Vky)

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
    main()





