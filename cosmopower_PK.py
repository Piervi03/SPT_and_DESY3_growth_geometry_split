import numpy as np
from math import sqrt as msqrt
from cosmopower import cosmopower_NN

from cosmosis.datablock import option_section


def setup(options):
    """Return redshift array, and the emulator object"""
    # Redshifts
    z_min_max = options.get_double_array_1d(option_section, 'z_min_max')
    N_z = options.get_int(option_section, 'N_z')
    z_arr = np.linspace(z_min_max[0], z_min_max[1], N_z)
    # Emulator
    restore_filename = options.get_string(option_section, 'restore_filename')
    emulator = cosmopower_NN(restore=True, restore_filename=restore_filename)
    return z_arr, emulator


def execute(block, setup_stuff):
    """Read cosmological parameters, run power spectrum emulator, and write to
    block."""
    # Setup
    z_arr, emulator = setup_stuff
    k_arr = emulator.modes
    # Parameters
    N = len(z_arr)
    h = block.get_double('cosmological_parameters', 'h0')
    params = {'omega_cdm': np.full(N, block.get_double('cosmological_parameters', 'omch2')),
              'omega_b': np.full(N, block.get_double('cosmological_parameters', 'ombh2')),
              'h': np.full(N, h),
              'n_s': np.full(N, block.get_double('cosmological_parameters', 'n_s')),
              'ln10^{10}A_s': np.full(N, block.get_double('cosmological_parameters', 'log1e10As')),
              'z': z_arr,
              }
    # k in units of h
    k = k_arr / h
    # Call the emulator
    Pk = emulator.ten_to_predictions_np(params) * h**3
    # Compute sigma_8
    kR = 8.*k
    window = 3. * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
    integrand_sigma2 = Pk[0] * window**2 * k**3
    sigma8_squ = .5/np.pi**2 * np.trapezoid(integrand_sigma2, np.log(k))
    # Write to block
    block.put_double('cosmological_parameters', 'sigma_8', msqrt(sigma8_squ))
    block.put_grid('cdm_baryon_power_lin', 'z', z_arr, 'k_h', k, 'p_k', Pk)
    return 0


def cleanup(config):
    pass
