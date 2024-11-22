# **TARANGPY**

**Tarang** is a pseudo-spectral solver designed for Hydrodynamic Turbulence. It leverages object-oriented programming and high-performance libraries like **NumPy** and **CuPy** for efficient computations, supporting both CPU and GPU environments. Equipped with a **Graphical User Interface (GUI)** and robust **post-processing** capabilities, Tarang simplifies running simulations and analyzing data, making it an ideal tool for studying turbulence.

Tarang can solve a variety of **Partial Differential Equations (PDEs)** in turbulent flows across both 2D and 3D domains:

- **Hydrodynamics**
- **Magnetohydrodynamics (MHD)**
- **Rayleigh-Bénard Convection**
- **Scalar field**

Additionally, Tarang supports various external forcing mechanisms including random modal forcing, bulk rotation and buoyancy.

The post-processing suite allows users to extract and analyze key simulation results using key quantities such as energy, dissipation, energy spectra, and flux for the fields involved.

**Note:** This is a limited sample version with only Hydrodynamics module without all the frills.

---

## **Installation and Usage**

Please ensure the following required packages are installed with Python 3.10 or higher for optimal performance:

- `numpy`
- `pyFFTW` (for CPU-based FFT)
- `cupy` (for GPU support)
- `h5py`

Make sure `cupy` is installed for the available cuda version. For example,

```bash
pip3 install numpy pyFFTW h5py cupy-cuda12x
```

Before running the code, configure the `para.py` file according to your requirements, and provide an initial `.h5` file containing the necessary field data. You can then execute the solver using either the CLI or GUI version.

**Creating Input Field**

To create the input field specify your required field in the file `pre_process\init_modes\init_hydro.py`

**Running the CLI Version**

```bash
python3 tarang.py
```

**Note:** Do not remove any variables from the `para.py` file. Replace `python3` with `python` as required.

---
