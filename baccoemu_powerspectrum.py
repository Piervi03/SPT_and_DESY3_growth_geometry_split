import numpy as np
import baccoemu

from cosmosis.datablock import option_section


def setup(options):
    """Return redshift array and the emulator object"""
    emulator = baccoemu.Matter_powerspectrum()
    z_min_max = options.get_double_array_1d(option_section, 'z_min_max')
    N_z = options.get_int(option_section, 'N_z')
    z_arr = np.linspace(z_min_max[0], z_min_max[1], N_z)
    return z_arr, emulator


def execute(block, stuff):
    """Read cosmological parameters, run power spectrum emulator, and write to
    block."""
    # Setup
    z_arr, emulator = stuff
    # Cosmology parameters
    cosmology = {
                 'omega_matter': block.get_double('cosmological_parameters', 'Omega_m'),
                 'omega_baryon': block.get_double('cosmological_parameters', 'Omega_b'),
                 'Omnuh2': block.get_double('cosmological_parameters', 'Omnuh2'),
                 'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
                 'ns': block.get_double('cosmological_parameters', 'n_s'),
                 'hubble': block.get_double('cosmological_parameters', 'h0'),
                 'sigma8': block.get_double('cosmological_parameters', 'sigma_8'),
                 'w0': block.get_double('cosmological_parameters', 'w'),
                 'wa': block.get_double('cosmological_parameters', 'wa')}
    cosmology['neutrino_mass'] = cosmology['Omnuh2']*94
    # Call the emulator
    tmp = np.empty((len(z_arr),2,200))
    for i,z in enumerate(z_arr):
        cosmology['expfactor'] = 1/(1+z)
        tmp[i] = emulator.get_linear_pk(cosmology)
        if not np.array_equal(tmp[i,0,:], tmp[0,0,:]):
            return 1
    # Write to block
    block.put_double_array_1d('matter_power_lin_cdm_baryon', 'z', z_arr)
    block.put_double_array_1d('matter_power_lin_cdm_baryon', 'k_h', tmp[0,0,:])
    block.put_double_array_nd('matter_power_lin_cdm_baryon', 'p_k', tmp[:,1,:])

    return 0


def cleanup(config):
    pass
