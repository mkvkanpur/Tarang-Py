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
from lib.global_fns.spectral_setup import *
from lib.global_fns.universal import Universal_arrays
import h5py

if para.device == "gpu":
      import cupy as ncp
      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()

      if para.gpu_direct_storage:
        import kvikio as kv
        import os
else:
      import numpy as ncp


def initial_hydro(U=VectorField()):
    ##read input field

    if (para.dimension == 2):
        Vkx = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=para.complex_dtype)
        Vkz = ncp.zeros((para.Nx,(para.Nz)//2+1), dtype=para.complex_dtype)
        Vky = []

    elif (para.dimension == 3):
        Vkx = ncp.zeros((para.Nx,para.Ny, para.Nz//2+1),dtype= para.complex_dtype)
        Vky = ncp.zeros((para.Nx,para.Ny, para.Nz//2+1),dtype= para.complex_dtype)
        Vkz = ncp.zeros((para.Nx,para.Ny, para.Nz//2+1),dtype= para.complex_dtype)

    if para.device == "gpu":
        with h5py.File(para.input_dir+"/"+para.input_file_name,'r') as f:
            Vkx = ncp.asarray(f['Vkx'][()])
            Vkz = ncp.asarray(f['Vkz'][()])

            Vkx, Vkz = reality_cond(Vkx), reality_cond(Vkz)
            if para.dimension == 3:
                Vky = ncp.asarray(f['Vky'][()])
                Vky = reality_cond(Vky)

    else:
        with h5py.File(para.input_dir+"/"+para.input_file_name,'r') as f:
            Vkx = f['Vkx'][()]
            Vkz = f['Vkz'][()]

            Vkx, Vkz = reality_cond(Vkx), reality_cond(Vkz)
            if para.dimension == 3:
                Vky = f['Vky'][()]
                Vky = reality_cond(Vky)

    #assign initial condition
    U.init_cond(Vkx, Vkz, Vky)

    #print(ncp.max(U.Vkx), ncp.max(U.Vky), ncp.max(U.Vkz))

    del(Vkx)
    del(Vky)
    del(Vkz)

    return

# this function is called in the loop
###
def output_hydro(t, U = VectorField(), univ=Universal_arrays()):

    if ((univ.iter > para.iter_field_save_start) and (univ.iter % (para.iter_field_save_inter) == 0)) or \
            (univ.t_last) or (univ.iter ==0):
        #print(univ.iter)
        if para.device == "gpu":
            univ.t_field_save.append(ncp.asnumpy(t))

            if para.gpu_direct_storage:
                os.makedirs(para.output_dir+"/fields" + "/Soln_%f" %(t), exist_ok = True)

                f = kv.CuFile(para.output_dir+"/fields" + "/Soln_%f/U.V1" %(t), "w")
                f.write(U.Vkx)
                f.close()

                f = kv.CuFile(para.output_dir+"/fields" + "/Soln_%f/U.V3" %(t), "w")
                f.write(U.Vkz)
                f.close()

                if (para.dimension == 3):
                    f = kv.CuFile(para.output_dir+"/fields" + "/Soln_%f/U.V2" %(t), "w")
                    f.write(U.Vky)
                    f.close()

            else:
                hf = h5py.File(para.output_dir+"/fields/Soln_%f.h5" %(t), 'w')
                hf.create_dataset('Vkx', data=ncp.asnumpy(U.Vkx))
                hf.create_dataset('Vkz', data=ncp.asnumpy(U.Vkz))
                if (para.dimension == 3):
                    hf.create_dataset('Vky', data=ncp.asnumpy(U.Vky))
                hf.close()

        else:
            univ.t_field_save.append(t)

            hf = h5py.File(para.output_dir+"/fields/Soln_%f.h5" %(t), 'w')
            hf.create_dataset('Vkx', data=U.Vkx)
            hf.create_dataset('Vkz', data=U.Vkz)
            if (para.dimension == 3):
                hf.create_dataset('Vky', data=U.Vky)
            hf.close()


    if ((univ.iter > para.iter_glob_energy_print_start) and \
            (univ.iter % (para.iter_glob_energy_print_inter) == 0)) or (univ.t_last) or (univ.iter ==0):

        if (para.FIXED_DT):
            print (t, univ.dt_fixed, U.compute_total_energy(), U.get_divergence(), para.nu*U.compute_dissipation())
        else:
            print (t, univ.dt, U.compute_total_energy(), U.get_divergence(), para.nu*U.compute_dissipation())

        if para.device == "gpu":
            univ.t_glob_energy_print.append(ncp.asnumpy(t))
            U.total_energy.append(ncp.asnumpy(U.compute_total_energy()))
            U.standard_dissipation.append(ncp.asnumpy(U.compute_dissipation()))

            
        else:
            univ.t_glob_energy_print.append(t)
            U.total_energy.append(U.compute_total_energy())
            U.standard_dissipation.append(U.compute_dissipation())

          

    if ((univ.iter > para.iter_modes_save_start) and \
            (univ.iter % (para.iter_modes_save_inter) == 0)) or (univ.t_last) or (univ.iter ==0):

        if para.device == "gpu":
            for k in range(ncp.shape(para.modes_save)[0]):

                U.Vkx_modes_t.append(ncp.asnumpy(U.Vkx[para.modes_save[k]]))
                U.Vkz_modes_t.append(ncp.asnumpy(U.Vkz[para.modes_save[k]]))

                if (para.dimension == 3):
                    U.Vky_modes_t.append(ncp.asnumpy(U.Vky[para.modes_save[k]]))


            univ.t_modes_save.append(ncp.asnumpy(t))

        else:
            for k in range(ncp.shape(para.modes_save)[0]):

                U.Vkx_modes_t.append(U.Vkx[para.modes_save[k]])
                U.Vkz_modes_t.append(U.Vkz[para.modes_save[k]])

                if (para.dimension == 3):
                    U.Vky_modes_t.append(U.Vky[para.modes_save[k]])

            univ.t_modes_save.append(t)

        U.Vkx_mode.append(U.Vkx_modes_t)
        U.Vkz_mode.append(U.Vkz_modes_t)

        U.Vkx_modes_t, U.Vkz_modes_t = [], []

        if (para.dimension == 3):
            U.Vky_mode.append(U.Vky_modes_t)

            U.Vky_modes_t = []

    return


def output_ekTk_hydro(t,  U = VectorField(), univ=Universal_arrays()):

    if ((univ.iter > para.iter_ekTk_save_start) and (univ.iter % (para.iter_ekTk_save_inter) == 0)) \
           or (univ.t_last) or (univ.iter ==0):

        U.compute_energy_spectrum_Tk(univ)

        if para.device == "gpu":
            univ.t_ekTk_save.append(ncp.asnumpy(t))

            U.ek_table.append(ncp.asnumpy(ncp.stack(U.ek).astype(None)))
            U.Tk_table.append(ncp.asnumpy(ncp.stack(U.Tk).astype(None)))

        else:
            univ.t_ekTk_save.append(t)

            U.ek_table.append(ncp.stack(U.ek).astype(None))
            U.Tk_table.append(ncp.stack(U.Tk).astype(None))

    return


def file_save_ekTk_hydro(U = VectorField(), univ=Universal_arrays()):

    hf2 = h5py.File(para.output_dir+"/ekTk.h5", 'w')
    hf2.create_dataset('t', data=univ.t_ekTk_save)
    hf2.create_dataset('k', data=k_array)
    hf2.create_dataset('ek', data=U.ek_table)
    hf2.create_dataset('Tk', data=U.Tk_table)
    hf2.close()

    return


def file_save_total_energy_hydro(U = VectorField(), univ=Universal_arrays()):

    hf2 = h5py.File(para.output_dir+"/glob.h5", 'w')
    hf2.create_dataset('t', data=univ.t_glob_energy_print)
    hf2.create_dataset('Eu', data=U.total_energy)
    hf2.create_dataset('dissipation_u', data=[i*para.nu for i in U.standard_dissipation])

    hf2.close()

    return


def file_save_t_field_save_hydro(univ=Universal_arrays()):

      hf2 = h5py.File(para.output_dir+"/t_field_save.h5", 'w')
      hf2.create_dataset('t', data=univ.t_field_save)
      hf2.close()

      return


def file_save_t_hydro(univ=Universal_arrays()):

      hf2 = h5py.File(para.output_dir+"/t.h5", 'w')
      hf2.create_dataset('t', data=univ.t)
      hf2.close()

      return


def file_save_modes_hydro(U = VectorField(), univ=Universal_arrays()):

      hf2 = h5py.File(para.output_dir+"/modes.h5", 'w')
      hf2.create_dataset('t', data=univ.t_modes_save)
      hf2.create_dataset('modes_ux', data=U.Vkx_mode)
      hf2.create_dataset('modes_uz', data=U.Vkz_mode)

      if (para.dimension == 3):
           hf2.create_dataset('modes_uy', data=U.Vky_mode)

      hf2.close()

      return


