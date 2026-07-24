"""
pre_bao.py

Precomputes everything bao_y6_like.py reads from the 'distances' section:
    z          - redshift grid
    d_m        - comoving distance, physical Mpc  
    h          - H(z)/c, in 1/Mpc                  
    rs_zdrag   - sound horizon at drag epoch, Mpc  

# The CMB measures the photon last-scattering surface at z_* ≈ 1090,
# where photons decouple from the baryon-photon plasma.
#
# BAO, however, traces the distribution of baryons and galaxies.
# The relevant epoch is the baryon drag epoch z_d, when photons stop
# efficiently dragging baryons through Thomson scattering.
#
# Therefore BAO analyses use the sound horizon at the drag epoch:
#     r_d = r_s(z_d)
#
# rather than the sound horizon at photon decoupling:
#     r_s(z_*)
#
# The difference is small:
#     r_s(z_*) ≈ 144 Mpc
#     r_s(z_d) ≈ 147 Mpc
"""
import numpy as np
from cosmosis.datablock import names, option_section
import cosmo


distances = names.distances

C_KM_S = 299792.458  # km/s


def setup(options):
    zmax = options.get_double(option_section, "zmax", default=1.2)
    nz = options.get_int(option_section, "nz", default=500)
    test_mode=options.get_bool(option_section, "test", default=False)#We can have a parameter to see if the modules works. Bool
    return {"zmax": zmax, "nz": nz,"test_mode": test_mode}


def _sound_horizon_eh98(block):
    """Eisenstein & Hu 1998 fitting formula for the drag-epoch sound
    horizon densities ombh2, omch2."""
    #We use omega_geo here to later calculate z_eq which is the time of matter radiation equality
    #The growth of structure is still negligible in that regime
    ombh2 = block['cosmological_parameters', 'ombh2']
    om_geo_h2 = block['cosmological_parameters', 'Omega_m_geo']*block['cosmological_parameters', 'h0']**2
    #Default is zero
    omnuh2 = block.get_double('cosmological_parameters', "omnuh2", 0.0)

    tcmb =2.7255

    #om_h2 = ombh2 + om_geo_h2 + omnuh2
    theta27 = tcmb / 2.7

    #Equilibrium of matter and radiation density. Find z out of that
    z_eq = 2.5e4 * om_geo_h2 * theta27 ** -4
    #same for k_eq. It's the mode that jsut crossed the horizon on that scale
    k_eq = 7.46e-2 * om_geo_h2 * theta27 ** -2  # Mpc^-1

    b1 = 0.313 * om_geo_h2 ** -0.419 * (1.0 + 0.607 * om_geo_h2 ** 0.674)
    b2 = 0.238 * om_geo_h2 ** 0.223
    z_drag = (1291.0 * om_geo_h2 ** 0.251 / (1.0 + 0.659 * om_geo_h2 ** 0.828)
              * (1.0 + b1 * ombh2 ** b2))
    
#R is the parameter in the plasma velocity. It depends on baryon density. c_s=c/sqrt(3*(1+R))
#Remember that R is defined as R=3/4* rho_b/rho_phot
    def R(z):
        return 31.5 * ombh2 * theta27 ** -4 * (z / 1.0e3) ** -1
    R_drag = R(z_drag)
    R_eq = R(z_eq)
    
#s is the comoving sound horizon distance r_d in Mpc according to Eisenstein & Hu (1998). 
    s = (2.0 / (3.0 * k_eq)) * np.sqrt(6.0 / R_eq) * np.log(
        (np.sqrt(1.0 + R_drag) + np.sqrt(R_drag + R_eq)) / (1.0 + np.sqrt(R_eq))
    )
    return s, z_drag

#This is intended as a test only
def _sound_horizon_eh98_test(cosmology):
    """Eisenstein & Hu 1998 fitting formula for the drag-epoch sound
    horizon densities ombh2, omch2."""
    #We use omega_geo here to later calculate z_eq which is the time of matter radiation equality
    #The growth of structure is still negligible in that regime
    ombh2 = cosmology['ombh2']
    om_geo_h2 = cosmology['Omega_m_geo']*cosmology['h0']**2
    #Default is zero
    omnuh2 = cosmology["omnuh2"]
    tcmb =2.7255

    #om_h2 = ombh2 + om_geo_h2 + omnuh2
    theta27 = tcmb / 2.7

    z_eq = 2.5e4 * om_geo_h2 * theta27 ** -4
    k_eq = 7.46e-2 * om_geo_h2 * theta27 ** -2  # Mpc^-1

    b1 = 0.313 * om_geo_h2 ** -0.419 * (1.0 + 0.607 * om_geo_h2 ** 0.674)
    b2 = 0.238 * om_geo_h2 ** 0.223
    z_drag = (1291.0 * om_geo_h2 ** 0.251 / (1.0 + 0.659 * om_geo_h2 ** 0.828)
              * (1.0 + b1 * ombh2 ** b2))
    
#R is the parameter in the plasma velocity. It depends on baryon density. c_s=c/sqrt(3*(1+R))
    def R(z):
        return 31.5 * ombh2 * theta27 ** -4 * (z / 1.0e3) ** -1

    R_drag = R(z_drag)
    R_eq = R(z_eq)
#s is the comoving sound horizon distance r_d in Mpc according to Eisenstein & Hu (1998). 
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
    h0 = block['cosmological_parameters', "h0"]  # H0 / 100

    cosmology = {
        "Omega_m_geo": omega_m_geo,
        "Omega_l":  1.0 - omega_m_geo,
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

    block[distances, "z"] = z
    block[distances, "d_m"] = D_M
    block[distances, "h"] = Hz_inv_Mpc

    # --- sound horizon ---
    rs_zdrag, z_drag = _sound_horizon_eh98(block)
    block[distances, "rs_zdrag"] = rs_zdrag
    block[distances, "zdrag"] = z_drag
    #Testing the module
    if test_mode==True:
        cosmology_test={
        'Omega_m_geo': 0.33,
        'Omega_l': 1.0 - omega_m_geo,
        'omnuh2': 0.0,
        'ombh2':0.022,
        'h0':0.67,
        'w0': -1,
        'wa': 0.0,
        }
        rs_zdrag_test, z_drag_test = _sound_horizon_eh98_test(cosmology_test)
        print("According to planck data the r_drag should be in Mpc: ",147)
        print("test rs_zdrag: ", rs_zdrag_test)
        print("test zdrag: ", z_drag_test)
        
    return 0


def cleanup(config):
    pass