# Copyright (c) 2024 Mahendra Verma. All rights reserved.

# Authors:
# - Mahendra Verma
# - Soumyadeep Chatterjee
# - Shashwat Nirgudkar

# This software is provided for non-commercial use only. You may not use, copy, modify, distribute, or otherwise exploit the software or any derivative works of the software for commercial purposes. Specifically, you may not sell, license, or profit from the software in any manner.

# Usage of the software is strictly limited to personal, academic, or non-commercial purposes. Any use, modification, or redistribution of this software outside of these constraints is prohibited without the explicit written consent of the copyright holder.

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
plt.rcParams['xtick.major.size'] = 0.7*4.0
plt.rcParams['xtick.major.width'] = 2*0.5
plt.rcParams['xtick.minor.size'] = 0.7*2.5
plt.rcParams['xtick.minor.width'] = 2*0.5
plt.rcParams['ytick.major.size'] = 0.7*4.0
plt.rcParams['ytick.major.width'] = 2*0.5
plt.rcParams['ytick.minor.size'] = 0.7*2.5
plt.rcParams['ytick.minor.width'] = 2*0.5
A=2*9.3#1.5*9.3
font = {'family' : 'serif', 'weight' : 'normal', 'size' : A}
plt.rc('font', **font)

fig, axes = plt.subplots(figsize = (12, 10))



#with h5py.File("glob.h5",'r') as f: 
#    t = np.asarray(f['t'][()])
#    E_u = np.asarray(f['E_u'][()])
#    E_b = np.asarray(f['E_b'][()])

    #eps_u = 1e-4*np.asarray(f['eps_u'][()])
    #eps_b = 3e-4*np.asarray(f['eps_b'][()])

'''
data = np.loadtxt("nohup_0.out", comments="%")

t_0= data[::2, 0]

energy_u_0 = data[::2, 2]

energy_b_0 = data[::2, 5]

tot_energy_0 = energy_u_0 + energy_b_0

cross_helicity_0 = data[::2, 8]

magnetic_potential_0 = data[1::2, 8]
'''

####

data = np.loadtxt("nohup.out", comments="%")

t= data[::2, 0]

energy_u = data[::2, 2]

energy_b = data[::2, 5]

tot_energy = energy_u + energy_b

cross_helicity = data[::2, 8]

magnetic_potential = data[1::2, 8]

'''
t = np.concatenate((t_0,t))

energy_u = np.concatenate((energy_u_0,energy_u))

energy_b = np.concatenate((energy_b_0,energy_b))

tot_energy = np.concatenate((tot_energy_0,tot_energy))

cross_helicity= np.concatenate((cross_helicity_0,cross_helicity))

magnetic_potential= np.concatenate((magnetic_potential_0,magnetic_potential))
'''
#plt.plot(t, energy_u/energy_b, 'r', label=r"$E_u$")

plt.plot(t, energy_u, 'r', label=r"$E_u$")

plt.plot(t, energy_b, 'b', label=r"$E_b$")

plt.plot(t, cross_helicity, 'k', label=r"$H_c$")

plt.plot(t, magnetic_potential, 'm', label=r"$E_A$")


plt.plot(t, energy_u+energy_b, 'g', label=r"$E$")

#plt.semilogy(t, energy_u/energy_b, 'b', label="E_b")


plt.xlabel(r"$t$")
plt.ylabel(r"$E$")

#plt.xlim(0, 150)
#plt.ylim(0, 3.0)


plt.legend(loc="best", ncol=3)
plt.tight_layout()
plt.savefig("total_energy_128.pdf")
plt.show()