# cosmosis_spt_cluster
The entire code is pure python and has no fancy dependencies.

## SPT cluster cosmology likelihood code for cosmosis
This package contains three main modules (in the cosmosis sense):
* compute_HMF computes the halo mass function on a grid of redshifts and masses
* abundance computes the likelihood of the cluster abundance (SPT-SZ SNR and redshifts)
* mass_calibration computes the likelihood using the clusters' follow-up mass calibration data

You can start by taking a look at the three `.yaml` files, and tell me to actually finish writing them.

## SPT-SZ mock generator
In addition, there is the standalone `mockgenerator.py` that creates realistic SPT-SZ mocks, including the split in different fields of different depths. Call it as `python mockgenerator.py mockinput.py` after setting all variables in the latter file.
