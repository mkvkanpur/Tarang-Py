# Copyright (c) 2024 Mahendra Verma. All rights reserved.

# Authors:
# - Mahendra Verma
# - Soumyadeep Chatterjee
# - Shashwat Nirgudkar

# This software is provided for non-commercial use only. You may not use, copy, modify, distribute, or otherwise exploit the software or any derivative works of the software for commercial purposes. Specifically, you may not sell, license, or profit from the software in any manner.

# Usage of the software is strictly limited to personal, academic, or non-commercial purposes. Any use, modification, or redistribution of this software outside of these constraints is prohibited without the explicit written consent of the copyright holder.

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




def tarang():

    import sys, os

    import para as para
    ################# Run pre proc script to initialize modes #################
    sys.path.append(para.para_directory +'/pre_process/init_modes/')

    ############# craeting input folder ########################
    os.makedirs(para.input_dir, exist_ok = True)
    #########################################################

    if para.kind == "HYDRO":
        import init_hydro
        init_hydro.init_hydro_main()

   
    ###########################################################################

    ################ Creating output folders ####################
        
    os.makedirs(para.output_dir, exist_ok = True)
    os.makedirs(para.output_dir+"/fields", exist_ok = True)
    ##############################################################

    if para.kind == "HYDRO":
        print("Problem kind is Hydro") 
        from src import main_hydro
        main_hydro.main_hydro()
        
if __name__ == "__main__":
    tarang()


    
