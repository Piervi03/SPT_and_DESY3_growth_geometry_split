"""
distances_geo_interface.py

CosmoSIS wrapper around the existing cosmo.py background-distance
functions (Ez, dA)

This module fills the standard 'distances' section (z, D_A, D_L) purely
from the geometric sector

Note on units: cosmo.dA returns distances in Mpc/h (DIST_H = c/100).

"""
import numpy as np
#The following is needed to invocke cosmosis section's names easily like: "cosmological_parameters"
from cosmosis.datablock import names, option_section
import cosmo  

cosmo_section = names.cosmological_parameters
distances = names.distances

#setup one is called just once 
#It reads the maximum redshift and the number of points
def setup(options):#options is the instance of the class. When we do options.something, we use a method of the class option. This is a default stuff
    zmax = options.get_double(option_section, "zmax", default=1.3)
    nz = options.get_int(option_section, "nz", default=300)
    return {"zmax": zmax, "nz": nz}

#execute is done everytime we use a new set of cosmological param during likelihood evaluation
#block and config are instances. config is the stuff that was passed from above.
def execute(block, config):
    zmax = config["zmax"]
    nz = config["nz"]
    omega_m_geo = block['cosmological_parameters', 'Omega_m_geo']

    cosmology = {
        "Omega_m_geo": omega_m_geo,
        "Omega_l": 1-omega_m_geo,
        "w0": block.get_double('cosmological_parameters', "w", -1.0),
        "wa": block.get_double('cosmological_parameters', "wa", 0.0),
    }
    #We use Da in physical units, not Mpc/h as it is normally from cosmo.dA
    z = np.linspace(0.0, zmax, nz)
    D_A = np.array([cosmo.dA(z_, cosmology) for z_ in z])/block['cosmological_parameters', 'h0']
    D_A[0] = 0.0  

    block[distances, "z"] = z
    block[distances, "D_A"] = D_A
    block[distances, "d_l"] = D_A * (1.0 + z) ** 2
    block[distances, "nz"] = len(z)

    return 0

#cleans memory and resources
def cleanup(config):
    pass
