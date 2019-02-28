from __future__ import division
import numpy as np
from cosmosis.datablock import option_section
import compute_HMF
import xarray as xr
import os

class EmptyClass:
    pass

def setup(options):
    # Print repository status
    path_to_repo = os.path.dirname(__file__)
    os.system("git --git-dir=%s/.git --work-tree=%s status"%(path_to_repo, path_to_repo))
    os.system("git --git-dir=%s/.git --work-tree=%s "%(path_to_repo, path_to_repo)+"show -s --format=%h")
    # Proceed with actual setup
    recalc_HMF = options.get_bool(option_section, 'recalc_HMF', default=True)
    save_HMF_to_disk = options.get_bool(option_section, 'save_HMF_to_disk', default=False)
    if recalc_HMF:
        HMF_calculator = compute_HMF.HMFCalculator(options)
    else:
        HMF_calculator = EmptyClass()
        HMF_calculator.HMF = xr.open_dataset('HMF.nc')
    HMF_calculator.recalc_HMF = recalc_HMF
    HMF_calculator.save_HMF_to_disk = save_HMF_to_disk
    return HMF_calculator

def execute(block, HMF_calculator):
    if HMF_calculator.recalc_HMF:
        HMF_calculator.compute_HMF(block)
        if HMF_calculator.save_HMF_to_disk:
            HMF = xr.DataArray(block.get_double_array_nd('HMF', 'dNdlnM'),
                               dims=['z', 'm'],
                               coords={'z': block.get_double_array_1d('HMF', 'z_arr'),
                                       'm': block.get_double_array_1d('HMF', 'M_arr')})
            HMF.to_netcdf('HMF.nc')

    else:
        block.put_double_array_1d('HMF', 'M_arr', np.array(HMF_calculator.HMF['m']))
        block.put_double_array_1d('HMF', 'z_arr', np.array(HMF_calculator.HMF['z']))
        block.put_double_array_nd('HMF', 'dNdlnM', np.array(HMF_calculator.HMF))
    return 0

def cleanup(config):
    pass
