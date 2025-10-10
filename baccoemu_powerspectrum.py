import numpy as np
from math import sqrt as msqrt
import baccoemu

from cosmosis.datablock import option_section


def setup(options):
    """Return redshift array and the emulator object"""
    emulator = baccoemu.Matter_powerspectrum()
    compute_sigma8 = options.get_bool(option_section, 'compute_sigma8', False)
    z_min_max = options.get_double_array_1d(option_section, 'z_min_max')
    N_z = options.get_int(option_section, 'N_z')
    z_arr = np.linspace(z_min_max[0], z_min_max[1], N_z)
    return compute_sigma8, z_arr, emulator


def execute(block, stuff):
    """Read cosmological parameters, run power spectrum emulator, and write to
    block."""
    # Setup
    compute_sigma8, z_arr, emulator = stuff
    # Parameters
    params = {'omega_matter': block.get_double('cosmological_parameters', 'Omega_m'),
              'omega_baryon': block.get_double('cosmological_parameters', 'Omega_b'),
              'neutrino_mass': block.get_double('cosmological_parameters', 'mnu'),
              'hubble': block.get_double('cosmological_parameters', 'h0'),
              'ns': block.get_double('cosmological_parameters', 'n_s'),
              'w0': block.get_double('cosmological_parameters', 'w'),
              'wa': block.get_double('cosmological_parameters', 'wa'),
              'A_s': block.get_double('cosmological_parameters', 'A_s'),
              }
    # Power spectrum P_{CDM+bar}(k)
    k, Pk = emulator.get_linear_pk(expfactor=1./(1.+z_arr),
                                   cold=True,
                                   **params)
    block.put_grid('cdm_baryon_power_lin', 'z', z_arr, 'k_h', k, 'p_k', Pk)
    # Compute sigma_8 for total matter
    if compute_sigma8:
        k_, Pk_ = emulator.get_linear_pk(expfactor=1.,
                                         cold=False,
                                         **params)
        kR = 8.*k_
        window = 3. * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
        integrand_sigma2 = Pk_ * window**2 * k_**3
        sigma8_squ = .5/np.pi**2 * np.trapezoid(integrand_sigma2, np.log(k_))
        block.put_double('cosmological_parameters', 'sigma_8', msqrt(sigma8_squ))
    return 0


def cleanup(config):
    pass
