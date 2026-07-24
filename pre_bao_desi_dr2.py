"""
pre_bao_desi_dr2.py

Precomputes everything desi_dr2.py reads from the 'distances' section:
    z          - redshift grid
    d_v        - angular-averaged BAO distance (D_V), physical Mpc
    d_m        - comoving distance, physical Mpc
    h          - H(z)/c, in 1/Mpc
    rs_zdrag   - sound horizon at drag epoch, Mpc

CHANGES relative to the DES-Y6-only version of this module:

  1. cosmological_parameters/H0 is now stored, since the
     DESI likelihood reads H0 directly rather than the reduced h0.
  2. Default zmax raised from 1.2 -> 2.5. DESI DR2's data sets span
     z_eff = 0.295 (BGS) up to z_eff = 2.330 (Lya).
    
"""
import numpy as np
from cosmosis.datablock import names, option_section
import cosmo


distances = names.distances

C_KM_S = 299792.458  # km/s


def setup(options):
    # DESI DR2's highest effective redshift is Lya at z_eff = 2.330, so the
    # default must comfortably exceed that (the DES-Y6-only version of this
    # module defaulted to 1.2, which is too low for DESI).
    zmax = options.get_double(option_section, "zmax", default=2.5)
    nz = options.get_int(option_section, "nz", default=500)
    test_mode = options.get_bool(option_section, "test", default=False)  # We can have a parameter to see if the modules works. Bool
    return {"zmax": zmax, "nz": nz, "test_mode": test_mode}


def _sound_horizon_eh98(block):
    """Eisenstein & Hu 1998 fitting formula for the drag-epoch sound
    horizon densities ombh2, omch2."""
    # We use omega_geo here to later calculate z_eq which is the time of matter radiation equality
    # The growth of structure is still negligible in that regime
    ombh2 = block['cosmological_parameters', 'ombh2']
    om_geo_h2 = block['cosmological_parameters', 'Omega_m_geo']*block['cosmological_parameters', 'h0']**2
    # Default is zero
    omnuh2 = block.get_double('cosmological_parameters', "omnuh2", 0.0)

    tcmb = 2.7255

    theta27 = tcmb / 2.7

    # Equilibrium of matter and radiation density. Find z out of that
    z_eq = 2.5e4 * om_geo_h2 * theta27 ** -4
    # same for k_eq. It's the mode that just crossed the horizon on that scale
    k_eq = 7.46e-2 * om_geo_h2 * theta27 ** -2  # Mpc^-1

    b1 = 0.313 * om_geo_h2 ** -0.419 * (1.0 + 0.607 * om_geo_h2 ** 0.674)
    b2 = 0.238 * om_geo_h2 ** 0.223
    z_drag = (1291.0 * om_geo_h2 ** 0.251 / (1.0 + 0.659 * om_geo_h2 ** 0.828)
              * (1.0 + b1 * ombh2 ** b2))

    # R is the parameter in the plasma velocity. It depends on baryon density. c_s=c/sqrt(3*(1+R))
    # Remember that R is defined as R=3/4* rho_b/rho_phot
    def R(z):
        return 31.5 * ombh2 * theta27 ** -4 * (z / 1.0e3) ** -1

    R_drag = R(z_drag)
    R_eq = R(z_eq)

    # s is the comoving sound horizon distance r_d in Mpc according to Eisenstein & Hu (1998).
    s = (2.0 / (3.0 * k_eq)) * np.sqrt(6.0 / R_eq) * np.log(
        (np.sqrt(1.0 + R_drag) + np.sqrt(R_drag + R_eq)) / (1.0 + np.sqrt(R_eq))
    )
    return s, z_drag


# This is intended as a test only
def _sound_horizon_eh98_test(cosmology):
    """Eisenstein & Hu 1998 fitting formula for the drag-epoch sound
    horizon densities ombh2, omch2."""
    ombh2 = cosmology['ombh2']
    om_geo_h2 = cosmology['Omega_m_geo']*cosmology['h0']**2
    omnuh2 = cosmology["omnuh2"]
    tcmb = 2.7255

    theta27 = tcmb / 2.7

    z_eq = 2.5e4 * om_geo_h2 * theta27 ** -4
    k_eq = 7.46e-2 * om_geo_h2 * theta27 ** -2  # Mpc^-1

    b1 = 0.313 * om_geo_h2 ** -0.419 * (1.0 + 0.607 * om_geo_h2 ** 0.674)
    b2 = 0.238 * om_geo_h2 ** 0.223
    z_drag = (1291.0 * om_geo_h2 ** 0.251 / (1.0 + 0.659 * om_geo_h2 ** 0.828)
              * (1.0 + b1 * ombh2 ** b2))

    def R(z):
        return 31.5 * ombh2 * theta27 ** -4 * (z / 1.0e3) ** -1

    R_drag = R(z_drag)
    R_eq = R(z_eq)
    s = (2.0 / (3.0 * k_eq)) * np.sqrt(6.0 / R_eq) * np.log(
        (np.sqrt(1.0 + R_drag) + np.sqrt(R_drag + R_eq)) / (1.0 + np.sqrt(R_eq))
    )
    return s, z_drag


def execute(block, config):
    zmax = config["zmax"]
    nz = config["nz"]
    test_mode = config["test_mode"]

    # --- d_m, h from omega_m_geo ---
    omega_m_geo = block['cosmological_parameters', "Omega_m_geo"]
    h0 = block['cosmological_parameters', "h0"]  # reduced Hubble constant, H0/100

    cosmology = {
        "Omega_m_geo": omega_m_geo,
        "Omega_l": 1.0 - omega_m_geo,
        "w0": block.get_double('cosmological_parameters', "w", -1.0),
        "wa": block.get_double('cosmological_parameters', "wa", 0.0),
    }

    z = np.linspace(0.0, zmax, nz)

    D_A_hMpc = np.array([cosmo.dA(z_, cosmology) for z_ in z])
    D_A_hMpc[0] = 0.0  # exact value at z=0
    D_A = D_A_hMpc / h0            # physical Mpc
    D_M = D_A * (1.0 + z)          # physical Mpc (flat, no curvature term)

    Ez = cosmo.Ez(z, cosmology)
    H0_phys = h0 * 100.0                    # km/s/Mpc
    Hz_inv_Mpc = (H0_phys * Ez) / C_KM_S     # 1/Mpc

    # --- angular-averaged BAO distance D_V(z), needed by desi_dr2.py 
    # D_V(z) = [ z * D_M(z)^2 * D_H(z) ]^(1/3),  with D_H(z) = c/H(z) = 1/Hz_inv_Mpc(z)
    D_V = np.zeros_like(z)
    D_V[1:] = (z[1:] * D_M[1:]**2 / Hz_inv_Mpc[1:])**(1.0 / 3.0)
    D_V[0] = 0.0  # exact value at z=0

    block[distances, "z"] = z
    block[distances, "d_m"] = D_M
    block[distances, "h"] = Hz_inv_Mpc
    block[distances, "d_v"] = D_V

    # --- sound horizon ---
    rs_zdrag, z_drag = _sound_horizon_eh98(block)
    block[distances, "rs_zdrag"] = rs_zdrag
    block[distances, "zdrag"] = z_drag

    # desi_dr2.py reads cosmological_parameters/H0 directly (full Hubble
    # constant, km/s/Mpc) to build H0*r_d. Only write it if it isn't
    # already there (e.g. from a consistency module earlier in the
    # pipeline), to avoid a duplicate-value error in the datablock.
    if not block.has_value('cosmological_parameters', 'H0'):
        block['cosmological_parameters', 'H0'] = H0_phys

    # Testing the module
    if test_mode == True:
        cosmology_test = {
            'Omega_m_geo': 0.33,
            'Omega_l': 1.0 - omega_m_geo,
            'omnuh2': 0.0,
            'ombh2': 0.022,
            'h0': 0.67,
            'w0': -1,
            'wa': 0.0,
        }
        rs_zdrag_test, z_drag_test = _sound_horizon_eh98_test(cosmology_test)
        print("According to planck data the r_drag should be in Mpc: ", 147)
        print("test rs_zdrag: ", rs_zdrag_test)
        print("test zdrag: ", z_drag_test)
    return 0


def cleanup(config):
    pass