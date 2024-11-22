# Copyright (c) 2024 Mahendra Verma. All rights reserved.

# Authors:
# - Mahendra Verma
# - Soumyadeep Chatterjee
# - Shashwat Nirgudkar

# This software is provided for non-commercial use only. You may not use, copy, modify, distribute, or otherwise exploit the software or any derivative works of the software for commercial purposes. Specifically, you may not sell, license, or profit from the software in any manner.

# Usage of the software is strictly limited to personal, academic, or non-commercial purposes. Any use, modification, or redistribution of this software outside of these constraints is prohibited without the explicit written consent of the copyright holder.

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




import numpy as np
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py as hp
import matplotlib.backends.backend_pdf

def plot_hydro(Path,plot_energy,plot_spectrum,plot_flux,input_time):
    print(input_time,type(input_time))
    sys.path.insert(0, Path)
    import para as para
    os.makedirs(Path+"/plots/", exist_ok=True)

    if(plot_energy):
        File_Name = 'glob.h5'
        dataset = hp.File(Path+'/'+File_Name, 'r')

        E_u = dataset['Eu'][:]
        T = dataset['t'][:]
        plt.clf()
        plt.loglog(T,E_u,label='Velocity_field_energy', color= 'blue', linewidth=2)
        plt.xlabel(r'$Time$')
        plt.ylabel(r'$Energy$')
        plt.title('Energy Plot')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path+'/plots/Energy_vs_time_hydro.pdf')

    if(plot_spectrum):

        File_Name = 'ekTk.h5'

        dataset = hp.File(Path+'/'+File_Name, 'r')

        T = dataset['t'][:]

        if len(input_time) == 2:
            time_index = np.where((T >= input_time[0]) & (T <= input_time[1]))

            print("Averaging between:", T[time_index[0][0]] ,"and", T[time_index[0][1]])
        else:

            time_index = np.where(T == input_time[0])

            if len(time_index) == 0:
                time_index = min(range(len(T)), key=lambda i: abs(T[i]-input_time[0]))
                print("Closest match:", T[time_index[0][0]])

        time_index = list(time_index[0])

        ek = dataset['ek'][time_index,1:,...]

        E_k = np.average(np.sum(ek, axis=-1), axis=0)

        k = np.arange(1,E_k.shape[0]+1,1)

        plt.clf()
        plt.loglog(k,E_k,label='Spectrum', color= 'red', linewidth=3)        ######## Plotting the spectrum data ##########
        plt.xlabel(r'$k$')          ######### Setting the X label ############
        plt.ylabel(r'$E(k)$')     ######### Setting the Y label ############
        plt.xlim(k[0],)
        plt.ylim(1e-10,1e2)      ######### Setting the y axis limits #######
        plt.title('Spectrum Plot')  ######### Title of the Plot ###############

        ################# Kolmogrov fit ( Please uncomment this section if you want to do komogrov fitting) #######################
        Kolmogrov_function=20*k**(-5.0/3.0)     ######### Creating kolmogrov line fit ########
        plt.loglog(k,Kolmogrov_function,'--',label='Kolmogrov_fit', color= 'green')   ######### Plotting kolmogrov line fit ########
        plt.legend()            ######### Enable the legend ############
        plt.tight_layout()
        plt.savefig(Path+'/plots/Spectrum_hydro.pdf')  ############## File name ###############

    if(plot_flux):
        File_Name = 'ekTk.h5'

        dataset = hp.File(Path+'/'+File_Name, 'r')

        T = dataset['t'][:]

        if len(input_time) == 2:
            time_index = np.where((T >= input_time[0]) & (T <= input_time[1]))

            print("Averaging between:", T[time_index[0][0]] ,"and", T[time_index[0][1]])
        else:

            time_index = np.where(T == input_time[0])

            if len(time_index) == 0:
                time_index = min(range(len(T)), key=lambda i: abs(T[i]-input_time[0]))
                print("Closest match:", T[time_index[0][0]])

        time_index = list(time_index[0])

        tk = dataset['Tk'][time_index,1:,:]

        T_k = np.average(np.sum((2*tk), axis = -1), axis=0)

        Flux = -np.cumsum(T_k)

        k= np.arange(1,T_k.shape[0]+1,1)

        plt.clf()
        plt.plot(k,Flux,label='Flux', color= 'red', linewidth=3)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$Flux$')
        plt.xlim(k[0],)
        plt.title('Flux Plot')
        plt.xscale('log')
        plt.yscale('symlog', linthresh=1E-3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(Path +'/plots/Flux_hydro.pdf')


    ################################################################################################
