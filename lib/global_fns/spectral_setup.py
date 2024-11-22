# Copyright (c) 2024 Mahendra Verma. All rights reserved.

# Authors:
# - Mahendra Verma
# - Soumyadeep Chatterjee
# - Shashwat Nirgudkar

# This software is provided for non-commercial use only. You may not use, copy, modify, distribute, or otherwise exploit the software or any derivative works of the software for commercial purposes. Specifically, you may not sell, license, or profit from the software in any manner.

# Usage of the software is strictly limited to personal, academic, or non-commercial purposes. Any use, modification, or redistribution of this software outside of these constraints is prohibited without the explicit written consent of the copyright holder.

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import para as para

if para.device == "gpu":
      import cupy as ncp

      dev1 = ncp.cuda.Device(para.device_rank)

      dev1.use()
else:
      import numpy as ncp
      import pyfftw.interfaces.numpy_fft as fp


## meshgrid

kx = ncp.arange(para.Nx)
kx[para.Nx//2+1:para.Nx] = kx[para.Nx//2+1:para.Nx]-para.Nx

ky = ncp.arange(para.Ny)
ky[para.Ny//2+1:para.Ny] = ky[para.Ny//2+1:para.Ny]-para.Ny

kz = ncp.linspace(0, para.Nz//2, para.Nz//2+1)

nax, nay, naz = kx, ky, kz

if not para.box_size_default:
    kx = para.kfactor[0]*kx
    ky = para.kfactor[1]*ky
    kz = para.kfactor[2]*kz


Ngrid = ncp.array([para.Nx, para.Ny, para.Nz])

if (para.dimension == 1):
    kz_mesh = kz
    ksqr = kz_mesh**2

    nsqr = naz**2
    dealiasing_index = ncp.where(nsqr > ncp.ceil((para.Nz**2)//3)+1)


if (para.dimension == 2):
    kx_mesh, kz_mesh = ncp.meshgrid(kx, kz, indexing = 'ij')
    ksqr = kx_mesh**2 + kz_mesh**2

    nax_mesh, naz_mesh = ncp.meshgrid(nax, naz, indexing = 'ij')
    nsqr = nax_mesh**2 + naz_mesh**2

    dealiasing_index = ncp.where(nsqr > ncp.ceil((min([para.Nx, para.Nz])**2)//3)+1)

    del nax_mesh, naz_mesh

if (para.dimension == 3):
    kx_mesh, ky_mesh, kz_mesh = ncp.meshgrid(kx, ky, kz, indexing = 'ij')
    ksqr = kx_mesh**2 + ky_mesh**2 + kz_mesh**2

    nax_mesh, nay_mesh, naz_mesh = ncp.meshgrid(nax, nay, naz, indexing = 'ij')
    nsqr = nax_mesh**2 + nay_mesh**2 + naz_mesh**2

    if (para.Ny == 1):
        dealiasing_index = ncp.where(nsqr > ncp.ceil((min([para.Nx, para.Nz])**2)//3)+1)
    else:
        dealiasing_index = ncp.where(nsqr > ncp.ceil((min(Ngrid)**2)//3)+1)

    del nax_mesh, nay_mesh, naz_mesh

del nsqr, nax, nay, naz


k_max = ncp.sqrt(ncp.sum(Ngrid * ncp.array(para.kfactor))**2)//2

min_radius_outside = int(ncp.ceil(ncp.sqrt(ncp.dot(Ngrid//2, Ngrid//2))))

k_min = k_max/(min_radius_outside+2)

k_array = (k_min*ncp.arange(0, min_radius_outside+1)).tolist()


def xderiv(Ak):
    return 1j*(kx_mesh*Ak)

def yderiv(Ak):
    return 1j*(ky_mesh*Ak)

def zderiv(Ak):
    return 1j*(kz_mesh*Ak)


def forward_transform(A, Ak):

    if (para.dimension == 1):
        if para.device == "gpu":
            Ak = (ncp.fft.rfft(A)/para.Nz)
        else:
            Ak = (fp.rfft(A)/para.Nz)
        return Ak

    elif (para.dimension == 2):
        if para.device == "gpu":
            Ak = (ncp.fft.rfft2(A)/(para.Nx*para.Nz))
        else:
            Ak = (fp.rfft2(A, threads=1)/(para.Nx*para.Nz))
        return Ak

    elif (para.dimension == 3):
        if para.device == "gpu":
            Ak = (ncp.fft.rfftn(A)/(para.Nx*para.Ny*para.Nz))
        else:
            Ak = (fp.rfftn(A)/(para.Nx*para.Ny*para.Nz))
        return Ak


def inverse_transform(Ak, A):

    if (para.dimension == 1):
        if para.device == "gpu":
            A = (ncp.fft.irfft(Ak)*para.Nz)
        else:
            A = (fp.irfft(Ak)*para.Nz)
        return A

    elif (para.dimension == 2):
        if para.device == "gpu":
            A = (ncp.fft.irfft2(Ak)*(para.Nx*para.Nz))
        else:
            A = (fp.irfft2(Ak, threads=1)*(para.Nx*para.Nz))
        return A

    elif (para.dimension == 3):
        if para.device == "gpu":
            A = (ncp.fft.irfftn(Ak)*(para.Nx*para.Ny*para.Nz))
        else:
            A = (fp.irfftn(Ak)*(para.Nx*para.Ny*para.Nz))
        return A


def dealias(Ak):
    Ak[dealiasing_index] = complex(0, 0)
    return Ak

def dealias_old(Ak):

    if (para.dimension == 1):
        Ak[para.Nz//3+1:para.Nz//2+1] = 0

    elif (para.dimension == 2):
        Ak[para.Nx//3+1:(2*para.Nx)//3+1, :] = 0
        Ak[:, para.Nz//3+1:para.Nz//2+1] = 0

    elif (para.dimension == 3):
        Ak[para.Nx//3+1:(2*para.Nx)//3+1, :, :] = 0
        Ak[:, para.Ny//3+1:(2*para.Ny)//3+1, :] = 0
        Ak[:, :, para.Nz//3+1:para.Nz//2+1] = 0

    return Ak


def reality_cond(Ak):

    if (para.dimension == 1):
        return Ak

    elif (para.dimension == 2):
        Ak[-1:para.Nx//2:-1,0] = ncp.conjugate(Ak[1:para.Nx//2,0])
        return Ak

    elif (para.dimension == 3):
        Ak[-1:para.Nx//2:-1,0,0] = ncp.conjugate(Ak[1:para.Nx//2,0,0]) #along x-axis
        Ak[0,para.Ny-1:para.Ny//2:-1,0] = ncp.conjugate(Ak[0,1:para.Ny//2,0]) #along y-axis

        #in x-y plane except x and y axis
        Ak[-1:para.Nx//2:-1,para.Ny-1:para.Ny//2:-1,0] = ncp.conjugate(Ak[1:para.Nx//2,1:para.Ny//2,0])
        Ak[-1:para.Nx//2:-1,para.Ny//2-1:0:-1,0] = ncp.conjugate(Ak[1:para.Nx//2,para.Ny//2+1:para.Ny,0])

        return Ak


def boundary_sin_cond(Ak):

    if(para.dimension==2):
        Ak[:,0].real = 0 #along x-axis
        Ak[-1:para.Nx//2:-1,:] = -(Ak[1:para.Nx//2,:]) #along x-axis
        Ak[0,:] = complex(0,0)
        Ak[para.Nx//2,:] = complex(0,0)
        return Ak


def boundary_cos_cond(Ak):

    if(para.dimension==2):
        Ak[:,0].imag = 0 #along x-axis
        Ak[-1:para.Nx//2:-1,:] = (Ak[1:para.Nx//2,:]) #along x-axis
        Ak[para.Nx//2,:].imag = 0
        return Ak


def craya_to_cartesian(u1, Vkx, Vkz, k_mag, i, j):
    if k_mag != 0:
        Vkx = ((j/k_mag)*u1)
        Vkz = ((-i/k_mag)*u1)
    return Vkx, Vkz


def compute_phi(i, j):
    phi = ncp.arctan2(j,i)
    return phi

