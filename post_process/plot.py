# Copyright (c) 2024 Mahendra Verma. All rights reserved.

# Authors:
# - Mahendra Verma
# - Soumyadeep Chatterjee
# - Shashwat Nirgudkar

# This software is provided for non-commercial use only. You may not use, copy, modify, distribute, or otherwise exploit the software or any derivative works of the software for commercial purposes. Specifically, you may not sell, license, or profit from the software in any manner.

# Usage of the software is strictly limited to personal, academic, or non-commercial purposes. Any use, modification, or redistribution of this software outside of these constraints is prohibited without the explicit written consent of the copyright holder.

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


######## Loading of Imp libraries ###########
import sys
#############################################
from output_folder_path import output_folder_path
############# Loading the system PAths correclty ###################
Path = output_folder_path #'/home/phyguest/Abhay/Tarang-py_workshop-master/output/'
sys.path.insert(0, Path)
import para as para
####################################################################

################# Change this accordingly ###################
#plot_type = "spectrum"          #### Specify plot as ---> SPECTRUM or ENERGY or FLUX
##############################################################

if(para.kind.lower() == "hydro"):
    import plot_hydro as plot_hydro
    pass


if(para.kind.lower() == "mhd"):
    import plot_MHD as plot_MHD
    pass

########## Under Development ###############

# if(para.kind.lower() == "scalar"):
#     import plot_scalar as plot_scalar
#     pass

# if(para.kind.lower() == "rbc"):
#     import plot_rbc as plot_rbc
#     pass

# if(para.kind.lower() == "chaos"):
#     import plot_chaos as plot_chaos
#     pass

#############################################
