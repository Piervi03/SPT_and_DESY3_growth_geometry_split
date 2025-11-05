import numpy as np

from cosmosis.datablock import option_section

import sigma8_z


def setup(options):
    """Only need to read redshift bins and centers from config file."""
    rescale = options.get_bool(option_section, 'rescale', default=False)
    zlim = options.get_double_array_1d(option_section, 'zlim')
    zmid = options.get_double_array_1d(option_section, 'zmid')
    return rescale, zlim, zmid


def execute(block, setup_stuff):
    """Replace power spectrum with rescaled version, and store array of
    sigma_8(z) for plotting."""
    # Input parameters
    rescale, zlim, zmid = setup_stuff
    if rescale:
        p_sigma8_z = np.array([block.get_double('cosmological_parameters', 'sigma8_z_%d' % i) for i in range(len(zmid))])
    else:
        p_sigma8_z = None
    # cdm+bar power spectrum (w/o neutrinos)
    z, k, Pk = block.get_grid('cdm_baryon_power_lin', 'z', 'k_h', 'p_k')
    # Rescale matter power spectrum
    Pk, z_out, sigma8_zout = sigma8_z.rescale_Pk(z, k, Pk, zlim, zmid, p_sigma8_z, rescale)
    # Write to cosmosis block
    if rescale:
        block.replace_double_array_nd('cdm_baryon_power_lin', 'p_k', Pk)
    for i in range(len(sigma8_zout)):
        block.put_double('cosmological_parameters', 'sigma8_z_out_%d' % i, sigma8_zout[i])
    return 0


def cleanup(config):
    pass
