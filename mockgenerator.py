from __future__ import division
import numpy as np
import os
import sys
import imp
from scipy.interpolate import RectBivariateSpline
from astropy.io import fits as pyfits
import pickle

import cosmo, Mconversion_concentration

##### Reference cosmology for which Mgas is measured
cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}

def main():
    ##### General setup
    # Input parameters and settings1
    configMod = imp.load_source('configMod', sys.argv[1])
    # SPT survey information
    SPTdata = imp.load_source('SPTdata', configMod.SPTdatafile)
    cosmology = configMod.cosmology
    scaling = configMod.scaling
    mcType = configMod.mcType
    np.random.seed(configMod.random_seed)
    Xray_obs = configMod.Xray_obs
    # Initialize c(M) calculator
    MCrel = Mconversion_concentration.ConcentrationConversion(configMod.mcType, cosmology)
    mass2obs = MassToObs(cosmology, scaling, configMod)

    cov = [[scaling['DWL_Megacam']**2, scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
        [scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
        [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]

    # Read HMF
    HMF = pickle.load(open('HMF.pkl', 'rb'))

    # Set up HMF interpolation
    dlnm = np.log(HMF['M_arr'][1]/HMF['M_arr'][0])
    HMF['dNdM_V'] = RectBivariateSpline(np.log(HMF['z_arr'][1:]), np.log(HMF['M_arr']), np.log(HMF['dNdlnM'][1:,:]*dlnm*(np.pi/180)**2))
    dz = .01
    z_arr = np.linspace(configMod.surveyCutRedshift[0], configMod.surveyCutRedshift[1], int((configMod.surveyCutRedshift[1]-configMod.surveyCutRedshift[0])/dz+1))
    dz = z_arr[1]-z_arr[0]

    # Get the mock catalog
    # The HMF is in units [Msun/h]
    mock, fieldnames = [], []
    xiArrEdge = np.linspace(5,10,101)
    xiArrBin = (xiArrEdge[1:]+xiArrEdge[:-1])/2
    dxi = xiArrEdge[1]-xiArrEdge[0]


    for fieldidx,field in enumerate(SPTdata.SPTfieldNames):
        mass2obs.thisSPTfieldCorrection = SPTdata.SPTfieldCorrection[SPTdata.SPTfieldNames.index(field)]

        massfunc = np.exp(HMF['dNdM_V'](np.log(z_arr), np.log(HMF['M_arr']))) * SPTdata.SPTfieldSize[fieldidx] * dz

        # Poisson realization
        N = np.random.poisson(massfunc)

        for i, z in enumerate(z_arr):
            for j, M in enumerate(HMF['M_arr']):
                if N[i,j]==0:
                    continue
                # draw (Mwl,Yx,zeta)|M
                obs_0 = [mass2obs(name, M, z) for name in ('WLMegacam', Xray_obs, 'zeta')]
                obs = np.exp(np.random.multivariate_normal(np.log(obs_0), cov, N[i,j]))
                for k in range(N[i,j]):
                    # Apply P(zeta)
                    if obs[k,2]>2.:
                        # draw xi|zeta
                        xi = np.random.normal(zeta2xi(obs[k,2]), scale=1.)
                        if xi>=configMod.surveyCutSZ[0]:
                            # Apply observational error to Mgas
                            Mg = np.random.lognormal(np.log(obs[k,1]), sigma=configMod.Xerr)
                            # Convert WL mass to 200c
                            M200h_WL = MCrel.MDelta_to_M200(obs[k,0],500.,z)

                            mock.append((M,z,xi,Mg,M200h_WL))
                            fieldnames.append(field)

        # False detections
        dNdxiFalse = SPTdata.SPTnFalse_beta[fieldidx] * SPTdata.SPTfieldSize[fieldidx]/2500. * SPTdata.SPTnFalse_alpha[fieldidx] * np.exp(-SPTdata.SPTnFalse_beta[fieldidx]*(xiArrBin-5.)) * dxi
        for i in range(len(dNdxiFalse)):
            N = np.random.poisson(dNdxiFalse[i])
            for k in range(N):
                mock.append((0.,0.,xiArrBin[i],0.,0.))
                fieldnames.append(field)


    mock = np.array(mock)


    ##### Select XVP
    nCluster = len(mock)
    print nCluster,'clusters'

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

        if configMod.profile_shape=='beta':
            # Build a beta profile with beta=2/3 (because that easy to integrate)
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
    names = []
    for i in range(nCluster):
        names.append('cluster'+str(i))
    names = np.array(names)
    XraySample = names[np.where(mock[:,3]!=0.)[0]]
    print XraySample

    # False detections
    redshiftLim = np.zeros(nCluster)
    redshiftLim[mock[:,1]==0.] = 1.4


    # M500 estimate, neede for defining X-ray observable
    M500_noh = np.random.lognormal(mock[:,0]/cosmology['h'], scaling['Dsz']/scaling['Bsz'])

    ##### Save catalog file
    # create numpy rec array
    names_arr = ['SPT_ID', 'field', 'xi', 'redshift', 'redshift_err', 'redshift_lim', 'Mg_MM', 'lnMg_err_MM', 'lnYx_err_MM', 'M_true', 'Tx_MM', 'M500', 'Mwl_200']
    format_arr = ['12a', '14a', 'f', 'f', 'f', 'f', '(2,80)f', 'f', 'f', 'f', 'f', 'f', 'f']
    data_arr = [names, fieldnames, mock[:,2], mock[:,1], np.zeros(nCluster), redshiftLim, Mgas, Xerrarr, Xerrarr, mock[:,0], 1e14*np.ones(nCluster), M500_noh, mock[:,4]]
    arr = np.rec.array(data_arr, names=names_arr, formats=format_arr)
    # Save to fits
    hdu = pyfits.BinTableHDU(data=arr)
    hdu.writeto('mockSPT2500d_'+sys.argv[1]+'.fits')



################################################################################

def zeta2xi(zeta):
    return (zeta**2 + 3)**.5


class MassToObs:
    def __init__(self, cosmology, scaling, configMod):
        self.cosmology = cosmology
        self.scaling = scaling
        self.thisSPTfieldCorrection = None
        self.SZmPivot = configMod.SZmPivot
        self.XraymPivot = configMod.XraymPivot
        self.YXPARAM = configMod.YXPARAM

    def __call__(self, name, mass, z):
        if name=='zeta':
            lnzeta = np.log(self.scaling['Asz']*self.thisSPTfieldCorrection)\
                + self.scaling['Bsz'] * np.log(mass/self.SZmPivot)\
                + self.scaling['Csz'] * np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))\
                + self.scaling['Esz'] * np.log(mass/self.SZmPivot)*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))
            return np.exp(lnzeta)
        elif name=='Yx':
            if self.YXPARAM=='SPT_XVP':
                return 3.*(mass*1e-14/(self.scaling['Ax'] * self.cosmology['h']**1.5
                    * (self.cosmology['h']/.72)**(2.5*self.scaling['Bx']-1.5)
                    * cosmo.Ez(z, self.cosmology)**self.scaling['Cx']))**(1/self.scaling['Bx'])
            elif self.YXPARAM=='Munich':
                return self.scaling['Ax']* (mass/self.cosmology['h']**1.5/self.XraymPivot)**self.scaling['Bx']\
                    * (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Cx']
        elif name=='Mgas':
            lnMgas = np.log(self.XraymPivot * self.scaling['Ax']) + self.scaling['Bx']*np.log(mass/self.XraymPivot/self.cosmology['h'])\
                + self.scaling['Cx']*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))\
                + self.scaling['Ex']*np.log(mass/self.XraymPivot/self.cosmology['h'])*np.log(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))
            return np.exp(lnMgas)
        elif name=='disp':
            h70z = self.cosmology['h']/.7*cosmo.Ez(z, self.cosmology)
            M200c = np.exp(lnM500_to_lnM200(z, np.log(mass)))
            if len(M200c)==1: M200c = M200c[0]
            return self.scaling['Adisp'] * (M200c/1e15/self.cosmology['h'])**(1/self.scaling['Bdisp']) * h70z**self.scaling['Cdisp']
        elif name=='richness':
            return self.scaling['Arichness'] * (mass/richmPivot)**self.scaling['Brichness']\
                * (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Crichness']
        elif name=='WLMegacam':
            return self.scaling['bWL_Megacam'] * mass
        elif name=='WLHST':
            return self.scaling['bWL_HST'] * mass
        elif name=='WLDES':
            return self.scaling['bWL_DES'] * mass
        else:
            raise ValueError("Observable not known:",name)


if __name__ == '__main__':
    main()
