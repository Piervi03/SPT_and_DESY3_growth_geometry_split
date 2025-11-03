import numpy as np
import os

from cosmosis.datablock import option_section

import compute_HMF_MiraTitan


class EmptyClass:
    pass


def setup(options):
    # Print repository status
    path_to_repo = os.path.dirname(__file__)
    os.system("git --git-dir=%s/.git --work-tree=%s status" % (path_to_repo, path_to_repo))
    os.system("git --git-dir=%s/.git --work-tree=%s " % (path_to_repo, path_to_repo)+"show -s --format=%h")
    # Proceed with actual setup
    Deltacrit = options.get_double(option_section, 'Deltacrit', default=500.)
    mcType = options.get_string(option_section, 'mcType')
    z = options.get_double_array_1d(option_section, 'z_arr')
    z_arr = np.linspace(z[0], z[1], int(z[2]))
    m = options.get_double_array_1d(option_section, 'M_arr')
    M_arr = np.logspace(m[0], m[1], int(m[2]))

    HMF_calculator = compute_HMF_MiraTitan.HMFCalculator(Deltacrit, mcType, z_arr, M_arr)

    return HMF_calculator


def execute(block, HMF_calculator):
    # Only need cosmo for E(z)-type stuff
    cosmology = {
        'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
        'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
        'Ommh2': block.get_double('cosmological_parameters', 'ommh2'),
        'Ombh2': block.get_double('cosmological_parameters', 'ombh2'),
        'Omnuh2': block.get_double('cosmological_parameters', 'omnuh2'),
        'h': block.get_double('cosmological_parameters', 'hubble')/100,
        'n_s': block.get_double('cosmological_parameters', 'n_s'),
        'sigma_8': block.get_double('cosmological_parameters', 'sigma_8'),
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa')}
    # Compute the HMF
    bad = HMF_calculator.compute_HMF(cosmology)
    if bad:
        return 1
    # Put it into block
    block.put_grid('HMF', 'z_arr', HMF_calculator.z_arr, 'M_arr', HMF_calculator.M_arr, 'dNdlnM', HMF_calculator.dNdlnM)
    block.put_double_array_nd('HMF', 'dNdlnM_unitVol', HMF_calculator.dNdlnM_unitVol)

    return 0


def cleanup(config):
    pass
