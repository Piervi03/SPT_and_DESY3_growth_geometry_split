"""
pre_sn_pantheon.py

CosmoSIS wrapper around the existing cosmo.py background-distance
functions (Ez, dA), supplying the 'distances' section (z, D_A, d_l)
for the Pantheon+SH0ES supernova likelihood using ONLY
omega_m_geo (the geometric/background sector).

Pantheon+ spans z=0.001-2.26 (much wider than DES-SN5YR's
~0.01-1.13)

We also need to include M as a parameter in the value file

"""
import numpy as np
from cosmosis.datablock import names, option_section
import cosmo

cosmo_section = names.cosmological_parameters
distances = names.distances


def setup(options):
    # Pantheon+ extends out to z=2.26 - default zmax raised accordingly
    zmax = options.get_double(option_section, "zmax", default=2.3)
    nz = options.get_int(option_section, "nz", default=500)
    return {"zmax": zmax, "nz": nz}


def execute(block, config):
    zmax = config["zmax"]
    nz = config["nz"]

    # geometric sector - FIXED: was reading Omega_m_growth
    omega_m_geo = block[cosmo_section, "omega_m_geo"]
    h0 = block[cosmo_section, "h0"]  # H0 / 100

    cosmology = {
        "Omega_m_geo": omega_m_geo,
        "Omega_l": block.get_double(cosmo_section, "omega_lambda", 1.0 - omega_m_geo),
        "w0": block.get_double(cosmo_section, "w", -1.0),
        "wa": block.get_double(cosmo_section, "wa", 0.0),
    }

    z = np.linspace(0.0, zmax, nz)

    # cosmo.dA returns Mpc/h; FIXED: divide by h0 for physical Mpc -
    # see docstring, this matters for Pantheon+SH0ES specifically
    D_A_hMpc = np.array([cosmo.dA(z_, cosmology) for z_ in z])
    D_A_hMpc[0] = 0.0
    D_A = D_A_hMpc / h0

    block[distances, "z"] = z
    block[distances, "D_A"] = D_A
    block[distances, "d_l"] = D_A * (1.0 + z) ** 2
    block[distances, "nz"] = len(z)

    return 0


def cleanup(config):
    pass