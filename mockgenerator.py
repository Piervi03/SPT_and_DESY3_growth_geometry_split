from __future__ import division, print_function
import numpy as np
import os
import sys
import time
import importlib
from scipy.interpolate import RectBivariateSpline
from astropy.table import Table

import compute_HMF_MiraTitan, cosmo, Mconversion_concentration, scaling_relations

##### Reference cosmology for which Mgas is measured
cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}

def main():
    ##### General setup
    # Input parameters and settings
    configMod = importlib.import_module(sys.argv[1][:-3])
    # SPT survey information
    SPT_survey = Table.read(configMod.SPT_survey, format='ascii.commented_header')
    cosmology = configMod.cosmology
    scaling = configMod.scaling
    rng = np.random.default_rng(configMod.random_seed)

    HMF = {'z': np.linspace(0, 2, 21),
           'm': np.logspace(13, 15.5, 251)}
    MiraTitan_HMF = compute_HMF_MiraTitan.HMFCalculator(200., 'Duffy08', HMF['z'], HMF['m'])
    bad = MiraTitan_HMF.compute_HMF(cosmology)
    if bad:
        print("Could not compute mass function")
    HMF['dNdlnM'] = MiraTitan_HMF.dNdlnM


    # Set up HMF interpolation
    dlnm = np.log(HMF['m'][1]/HMF['m'][0])
    HMF_dNdM_V = RectBivariateSpline(np.log(HMF['z'][1:]), np.log(HMF['m']), np.log(HMF['dNdlnM'][1:,:]*dlnm*(np.pi/180)**2), kx=1, ky=1)
    dz = .01
    z_arr = np.linspace(configMod.surveyCutRedshift[0], configMod.surveyCutRedshift[1], int((configMod.surveyCutRedshift[1]-configMod.surveyCutRedshift[0])/dz) + 1)
    dz = z_arr[1]-z_arr[0]

    # [DES WL, X-ray, SZ, richness, HST WL]
    covs = np.empty((len(z_arr), len(HMF['m']), 5, 5))
    for i, z in enumerate(z_arr):
        for j, M in enumerate(HMF['m']):
            scaling['DWL_DES'] = scaling_relations.WLscatter('main', M, z, scaling)
            covs[i,j,:,:] = [[scaling['DWL_DES']**2, scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['rhoWLrichness']*scaling['DWL_DES']*scaling['Drichness'], 0],
                             [scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['rhoXrichness']*scaling['Dx']*scaling['Drichness'], scaling['rhoWLX']*scaling['DWL_HST']*scaling['Dx']],
                             [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_HST']],
                             [scaling['rhoWLrichness']*scaling['DWL_DES']*scaling['Drichness'], scaling['rhoXrichness']*scaling['Dx']*scaling['Drichness'], scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Drichness']**2, scaling['rhoWLrichness']*scaling['DWL_HST']*scaling['Drichness']],
                             [0, scaling['rhoWLX']*scaling['DWL_HST']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_HST'], scaling['rhoWLrichness']*scaling['DWL_HST']*scaling['Drichness'], scaling['DWL_HST']**2]]


    # Get the mock catalog
    # The HMF is in units [Msun/h]
    mock, fieldnames = [], []
    xiArrEdge = np.linspace(5,10,101)
    xiArrBin = (xiArrEdge[1:]+xiArrEdge[:-1])/2
    dxi = xiArrEdge[1]-xiArrEdge[0]


    for fieldidx,field in enumerate(SPT_survey['FIELD']):
        print(field, fieldidx, 'out of %d'%len(SPT_survey['FIELD']))
        massfunc = np.exp(HMF_dNdM_V(np.log(z_arr), np.log(HMF['m']))) * SPT_survey['AREA'][fieldidx] * dz

        # Poisson realization
        N = rng.poisson(massfunc)

        obs_0 = np.array([scaling_relations.mass2obs(name, HMF['m'][None,:], z_arr[:,None], scaling, cosmology)
                          for name in ('WLDES', configMod.Xray_obs, 'zeta', 'richness',)])
        obs_0 = np.concatenate((obs_0, HMF['m']*np.ones((len(z_arr), len(HMF['m'])))[None,:]))

        # Field depth
        obs_0[2,:]*= SPT_survey['GAMMA'][fieldidx]


        for i, z in enumerate(z_arr):
            for j, M in enumerate(HMF['m']):
                if N[i,j]==0:
                    continue

                obs = np.exp(rng.multivariate_normal(np.log(obs_0[:,i,j]), covs[i,j], N[i,j]))

                keep = (obs[:,2]>scaling['zeta_min']).nonzero()[0]
                for k in keep:
                    # draw xi|zeta
                    xi = rng.normal(scaling_relations.zeta2xi(obs[k,2]), scale=1.)
                    if xi>=SPT_survey['XI_MIN'][fieldidx]:
                        # Apply observational error to Mgas
                        Mg = rng.lognormal(np.log(obs[k,1]), sigma=configMod.Xerr)

                        # Observed richness
                        if configMod.richness_scatter_model=='lognormal':
                            richness_obs = obs[k,3]
                        elif configMod.richness_scatter_model=='lognormalrelPoisson':
                            richness_obs = np.exp(rng.normal(np.log(obs[k,3]), scale=1/np.sqrt(obs[k,3])))
                        elif configMod.richness_scatter_model=='lognormalGaussPoisson':
                            richness_obs = rng.normal(obs[k,3], scale=np.sqrt(obs[k,3]))
                        else:
                            raise ValueError("Unknown value for richness_scatter_model")
                        mock.append((M, z, xi, Mg, obs[k,0], richness_obs, obs[k,4]))
                        fieldnames.append(field)

        # False detections
        # dNdxiFalse = SPT_survey['BETA'][fieldidx] * SPT_survey['AREA'][fieldidx]/2500. * SPT_survey['ALPHA'][fieldidx] * np.exp(-SPT_survey['BETA'][fieldidx]*(xiArrBin-5.)) * dxi
        # for i in range(len(dNdxiFalse)):
        #     N = rng.poisson(dNdxiFalse[i])
        #     for k in range(N):
        #         mock.append((0., 0., xiArrBin[i], 0., 0., 0.))
        #         fieldnames.append(field)

    names = np.array(['cluster%d'%i for i in range(len(mock))])

    mock = np.array(mock)

    ##### HST weak lensing
    HST_z_range = ((mock[:,1]>.6)&(mock[:,1]<1.1)).nonzero()[0]
    HST_idx = np.argsort(mock[HST_z_range,2])[-30:]
    # HST_idx = rng.choice(HST_z_range, 30, replace=False)
    mask = np.ones(len(mock), np.bool)
    mask[HST_z_range[HST_idx]] = 0
    mock[mask,-1] = 0.

    ##### Select XVP
    nCluster = len(mock)
    print(nCluster,'clusters', 'xi max', np.amax(mock[:,2]), 'lambda min', np.amin(mock[:,5]))

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
            rcFactor = 3 + 4*rng.random()
            rc = r500/rcFactor
            Micm = rc**2 * (rArr - rc*np.arctan(rArr/rc))
            # Normalize it such that Micm(r500) = mock[i,3]
            Micmr500 = np.interp(r500, rArr, Micm)
            Micm*= mock[i,3]/Micmr500
        elif configMod.profile_shape=='PL':
            Micm = mock[i,3] * (rArr/r500)**(scaling['slope_MgR'] + scaling['slope_MgR_std']*rng.standard_normal())

        # Scale back to ref cosmology
        Micmref = Micm * (dAref/dA)**2.5

        Mgas[i][0] = r_ref
        Mgas[i][1] = Micmref


    ##### Bookkeeping of names
    XraySample = names[np.where(mock[:,3]!=0.)[0]]
    # print(XraySample)

    # False detections
    redshiftLim = np.zeros(nCluster)
    redshiftLim[mock[:,1]==0.] = 1.4

    # M500 estimate, neede for defining X-ray observable
    M500_noh = rng.lognormal(np.log(mock[:,0]/cosmology['h']), scaling['Dsz']/scaling['Bsz'])

    # Theta_core (random)
    theta_core_Mpc = rng.exponential(scale=1/3.76, size=len(mock))
    theta_core_arcmin = theta_core_Mpc*cosmology['h'] / [cosmo.dA(z, cosmology) for z in mock[:,1]] * 180/np.pi * 60
    theta_core = np.round(theta_core_arcmin*4)/4
    theta_core[theta_core>3] = 3.

    ##### Save catalog file
    names_arr = ['SPT_ID', 'FIELD', 'XI', 'THETA_CORE', 'REDSHIFT', 'REDSHIFT_UNC', 'REDSHIFT_LIMIT',
                 'Mg_MM', 'lnMg_err_MM', 'lnYx_err_MM',
                 'M_true', 'Tx_MM',
                 'M500',
                 'Mwl_DES_200', 'Mwl_HST_200',
                 'richness',]
    data_arr = [names, fieldnames, mock[:,2], theta_core, mock[:,1], np.zeros(nCluster), redshiftLim,
                Mgas, Xerrarr, Xerrarr,
                mock[:,0], 1e14*np.ones(nCluster), M500_noh,
                mock[:,4], mock[:,6],
                mock[:,5],]
    # Save to fits
    cat = Table(data_arr, names=names_arr)
    cat.write('mock_%s.fits'%time.strftime("%y%m%d-%H%M%S"))



if __name__ == '__main__':
    main()
