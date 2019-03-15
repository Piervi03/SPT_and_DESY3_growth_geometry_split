from __future__ import division, print_function
import numpy as np
import os
import sys
import imp
from scipy.interpolate import RectBivariateSpline
from astropy.io import fits as pyfits
from astropy.table import Table
import xarray as xr

import cosmo, Mconversion_concentration, scaling_relations

##### Reference cosmology for which Mgas is measured
cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}

def main():
    ##### General setup
    # Input parameters and settings1
    configMod = imp.load_source('configMod', sys.argv[1])
    # SPT survey information
    SPT_survey = Table.read('data/SPT_survey.txt', format='ascii.commented_header')
    cosmology = configMod.cosmology
    scaling = configMod.scaling
    mcType = configMod.mcType
    np.random.seed(configMod.random_seed)
    Xray_obs = configMod.Xray_obs
    # Initialize c(M) calculator
    MCrel = Mconversion_concentration.ConcentrationConversion(configMod.mcType, cosmology)

    # [WL, X-ray, SZ, richness]
    cov = [[scaling['DWL_Megacam']**2, scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoWLrichness']*scaling['DWL_Megacam']*scaling['Drichness']],
        [scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['rhoXrichness']*scaling['Dx']*scaling['Drichness']],
        [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness']],
        [scaling['rhoWLrichness']*scaling['DWL_Megacam']*scaling['Drichness'], scaling['rhoXrichness']*scaling['Dx']*scaling['Drichness'], scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Drichness']**2]]

    # Read HMF
    HMF = xr.open_dataset('HMF.nc')

    # Set up HMF interpolation
    dlnm = np.log(HMF['m'][1]/HMF['m'][0])
    HMF_dNdM_V = RectBivariateSpline(np.log(HMF['z'][1:]), np.log(HMF['m']), np.log(HMF.to_array()[0][1:,:]*dlnm*(np.pi/180)**2))
    dz = .01
    z_arr = np.linspace(configMod.surveyCutRedshift[0], configMod.surveyCutRedshift[1], int((configMod.surveyCutRedshift[1]-configMod.surveyCutRedshift[0])/dz+1))
    dz = z_arr[1]-z_arr[0]

    # Get the mock catalog
    # The HMF is in units [Msun/h]
    mock, fieldnames = [], []
    xiArrEdge = np.linspace(5,10,101)
    xiArrBin = (xiArrEdge[1:]+xiArrEdge[:-1])/2
    dxi = xiArrEdge[1]-xiArrEdge[0]


    for fieldidx,field in enumerate(SPT_survey['FIELD']):

        massfunc = np.exp(HMF_dNdM_V(np.log(z_arr), np.log(HMF['m']))) * SPT_survey['AREA'][fieldidx] * dz

        # Poisson realization
        N = np.random.poisson(massfunc)

        for i, z in enumerate(z_arr):
            for j, M in enumerate(HMF['m']):
                if N[i,j]==0:
                    continue
                # draw (Mwl,Yx,zeta,richness)|M
                obs_0 = [scaling_relations.mass2obs(name, M, z, scaling, cosmology)
                         for name in ('WLMegacam', Xray_obs, 'zeta', 'richness')]
                obs_0[2]*= SPT_survey['GAMMA'][fieldidx]
                obs = np.exp(np.random.multivariate_normal(np.log(obs_0), cov, N[i,j]))
                for k in range(N[i,j]):
                    # Apply P(zeta)
                    if obs[k,2]>2.:
                        # draw xi|zeta
                        xi = np.random.normal(scaling_relations.zeta2xi(obs[k,2]), scale=1.)
                        if xi>=configMod.surveyCutSZ[0]:
                            # Apply observational error to Mgas
                            Mg = np.random.lognormal(np.log(obs[k,1]), sigma=configMod.Xerr)
                            # Convert WL mass to 200c
                            M200h_WL = MCrel.MDelta_to_M200(obs[k,0],500.,z)
                            # Observed richness
                            richness_int = np.random.lognormal(np.log(obs[k,3]), obs[k,3]**-.5)
                            richness_obs = np.random.normal(richness_int, configMod.richness_err)

                            mock.append((M, z, xi, Mg, M200h_WL, richness_obs))
                            fieldnames.append(field)

        # False detections
        dNdxiFalse = SPT_survey['BETA'][fieldidx] * SPT_survey['AREA'][fieldidx]/2500. * SPT_survey['ALPHA'][fieldidx] * np.exp(-SPT_survey['BETA'][fieldidx]*(xiArrBin-5.)) * dxi
        for i in range(len(dNdxiFalse)):
            N = np.random.poisson(dNdxiFalse[i])
            for k in range(N):
                mock.append((0., 0., xiArrBin[i], 0., 0., 0.))
                fieldnames.append(field)


    mock = np.array(mock)


    ##### Select XVP
    nCluster = len(mock)
    print(nCluster,'clusters')

    # Select nXrayCluster highest xi for Yx follow-up
    XVP = np.argsort(mock[:,2])
    mock[XVP[:-configMod.nXrayCluster],3] = 0.

    Xerrarr = configMod.Xerr*np.ones(nCluster)
    Xerrarr[XVP[:-configMod.nXrayCluster]] = 0.


    ##### Create X-ray gas mass profiles
    # For maximal confusion, this part is in decent units, with factors of h
    # because Xrayprofile.py is in nice units as well :)
    Mgas = np.zeros((nCluster,2,80))
    r_ref = np.linspace(25, 2000, 80)
    for i in range(nCluster):
        if mock[i,3] == 0.:
            continue

        zClust = mock[i,1]

        # Angular diameter distances
        # The reference cosmology matches Mike M's choice for the XVP data
        dAref = cosmo.dA(zClust, cosmologyRef) / cosmologyRef['h']
        dA = cosmo.dA(zClust, cosmology) / cosmology['h']

        # Scale r_ref to current cosmo
        rArr = r_ref * dA/dAref

        # Get the true r500
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(zClust, cosmology)**2.
        r500 = 1000 * (3*mock[i,0]/(4*np.pi*500*rho_c_z))**(1/3)
        r500/= cosmology['h']

        if configMod.profile_shape=='BETA':
            # Build a BETA profile with BETA=2/3 (because that easy to integrate)
            # and random r500/7 < rc < r500/3
            # Note that the amplitude doesn't matter because this thing gets rescaled below
            rcFactor = 3 + 4*np.random.random()
            rc = r500/rcFactor
            Micm = rc**2 * (rArr - rc*np.arctan(rArr/rc))
            # Normalize it such that Micm(r500) = mock[i,3]
            Micmr500 = np.interp(r500, rArr, Micm)
            Micm*= mock[i,3]/Micmr500
        elif configMod.profile_shape=='PL':
            Micm = mock[i,3] * (rArr/r500)**(scaling['slope_MgR'] + scaling['slope_MgR_std']*np.random.randn())

        # Scale back to ref cosmology
        Micmref = Micm * (dAref/dA)**2.5

        Mgas[i][0] = r_ref
        Mgas[i][1] = Micmref


    ##### Bookkeeping of names
    names = np.array(['cluster%d'%i for i in range(nCluster)])
    XraySample = names[np.where(mock[:,3]!=0.)[0]]
    print(XraySample)

    # False detections
    redshiftLim = np.zeros(nCluster)
    redshiftLim[mock[:,1]==0.] = 1.4

    # M500 estimate, neede for defining X-ray observable
    M500_noh = np.random.lognormal(np.log(mock[:,0]/cosmology['h']), scaling['Dsz']/scaling['Bsz'])

    ##### Save catalog file
    # create numpy rec array
    names_arr = ['SPT_ID', 'FIELD', 'XI', 'REDSHIFT', 'REDSHIFT_UNC', 'REDSHIFT_LIM',
                 'Mg_MM', 'lnMg_err_MM', 'lnYx_err_MM',
                 'M_true', 'Tx_MM', 'M500', 'Mwl_200',
                 'richness', 'richness_err']
    format_arr = ['12a', '14a', 'f', 'f', 'f', 'f', '(2,80)f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f']
    data_arr = [names, fieldnames, mock[:,2], mock[:,1], np.zeros(nCluster), redshiftLim,
                Mgas, Xerrarr, Xerrarr, mock[:,0], 1e14*np.ones(nCluster),
                M500_noh, mock[:,4],
                mock[:,5], configMod.richness_err*np.ones(nCluster)]
    arr = np.rec.array(data_arr, names=names_arr, formats=format_arr)
    # Save to fits
    hdu = pyfits.BinTableHDU(data=arr)
    hdu.writeto('mockSPT2500d_'+sys.argv[1]+'.fits')



if __name__ == '__main__':
    main()
