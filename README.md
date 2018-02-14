# cosmosis_spt_cluster
The entire code is pure python and has no fancy dependencies.

## SPT cluster cosmology likelihood code for cosmosis
This package contains three main modules (in the cosmosis sense):
* compute_HMF computes the halo mass function on a grid of redshifts and masses
* abundance computes the likelihood of the cluster abundance (SPT-SZ SNR and redshifts)
* mass_calibration computes the likelihood using the clusters' follow-up mass calibration data

You can start by taking a look at the three `.yaml` files, and tell me to actually finish writing them.

## Impact of neutrinos on halo mass function (talk to Matteo!)
`compute_HMF.py` uses the mean density of CDM+baryons (no neutrinos). To get the linear matter power spectrum from CAMB, you want to modify `cosmosis-standard-library/boltzmann/camb/camb_interface.f90:440` from

`call Transfer_GetMatterPowerData(MT, PK, 1)`

to

`call Transfer_GetMatterPowerData(MT, PK, 1, var1=Transfer_nonu, var2=Transfer_nonu)`

and re-compile.

## SPT-SZ mock generator
In addition, there is the standalone `mockgenerator.py` that creates realistic SPT-SZ mocks, including the split in different fields of different depths. Call it as `python mockgenerator.py mockinput.py` after setting all variables in the latter file.
