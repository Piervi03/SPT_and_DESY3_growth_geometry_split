from __future__ import division
import numpy as np
from numpy.lib import scimath as sm
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
import h5py
import imp
import os

import cosmo, miscentering

########################################
##### This class reads and stores shear data and calculates P(shear|P(M))
class SPTlensing:

    def __init__(self, catalog, WLsimcalibfile, HSTfile, MegacamFile, DESfile, DES_betabias_file):
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration
        # beta bias redshift-interpolation for DES
        data_ = np.loadtxt(DES_betabias_file, unpack=True)[:3]
        self.DES_betabias_mean = interp1d(data_[1], data_[0], kind='cubic')
        self.DES_betabias_var = interp1d(data_[1], data_[2]**2, kind='cubic')
        # Miscentering
        self.DES_miscenterer = miscentering.MisCentering(kind=self.WLcalib['DES_miscenter_kind'])

        readdata(catalog, HSTfile, MegacamFile, DESfile)


    ########################################
    # Get P(Mwl) from dP/dMwl and shear data
    def like(self, data, mArr, cosmology, MCrel, scaling):
        """Return likelihood of shear profile for a given cluster (index) given
        an array of cluster masses."""
        self.zcluster = data['REDSHIFT']
        self.WLdata = data['WLdata']

        ##### Miscentering parameters
        # if self.WLcalib['DES_miscenter_kind']=='SPT':
        #     self.DES_miscenterer.SPT_kappa = scaling['SPT_kappa']

        ##### Cosmology and halo stuff, in 1/h units
        self.get_beta(cosmology)
        rho_c_z, Dl, delta_c, r_s = self.get_cluster_properties(mArr, MCrel, cosmology)

        ##### Likelihood
        if self.WLdata['datatype']=='DES':
            pOfMass = self.like_DES(rho_c_z, Dl, delta_c, r_s, data['XI'], scaling)
        elif self.WLdata['datatype']=='Megacam':
            pOfMass = self.like_Megacam(rho_c_z, Dl, delta_c, r_s)
        elif self.WLdata['datatype']=='HST':
            pOfMass = self.like_HST(rho_c_z, Dl, delta_c, r_s)

        return pOfMass



    ########################################

    def get_cluster_properties(self, mArr, MCrel, cosmology):
        """Return rho_c(z_cluster), luminosity distance (z_cluster), delta_c,
        and r_s."""
        Dl = cosmo.dA(self.zcluster, cosmology)
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.zcluster, cosmology)**2 # [h^2 Msun/Mpc^3]
        M200c = np.exp(MCrel.lnM_to_lnM200(self.zcluster, np.log(mArr)))[0]
        r200c = (3*M200c/4/np.pi/200/rho_c_z)**(1/3)
        c200c = MCrel.calC200(M200c, self.zcluster)
        delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
        r_s = r200c/c200c
        return rho_c_z, Dl, delta_c, r_s


    ########################################

    def like_DES(self, rho_c_z, Dl, delta_c, r_s, xi, scaling):
        """Return array P(DES data|Mwl). Note that this is not normalized wrt
        the mArr for a good reason: In general, the mArr will not cover the full
        pOfMass range, and it varies as a function of SZ parameters. However,
        pOfMass is a product of normalized distributions, and so its
        normalization is constant throughout parameter space."""
        r_deg_interp = np.logspace(-3, np.log10(1.2*self.WLdata['r_deg'][-1]), 16)
        # Dimensionless radial distance [Radius][Mass]
        x = r_deg_interp[:,None] * Dl * np.pi/180 / r_s[None,:]
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/Dl/self.beta_avg

        # Realization of shear and beta bias
        betabias_mean_ = self.DES_betabias_mean(self.zcluster)
        betabias_var_ = self.DES_betabias_var(self.zcluster)
        total_var_ = betabias_var_ + self.WLcalib['DESshearErr']**2 + self.WLcalib['DEScontamCorr']**2
        dev_ = betabias_mean_ + scaling['DESbias'] * np.sqrt(total_var_)
        Sigma_c*= dev_
        # gamma_t, kappa, g_t [Radius][Mass]
        gamma_2d = get_Delta_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        kappa_2d = get_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        g_2d = gamma_2d/(1-kappa_2d)

        pOfMass = np.empty(g_2d.shape[1])
        for i in range(len(pOfMass)):
            g_2d[:,i], cov_miscenter_ = self.DES_miscenterer.get_profile_mean_cov(r_deg_interp, g_2d[:,i], .2,
                                                                                  SPT_xi=xi,
                                                                                  SPT_thetac=1)#data['THETA_CORE'])
            g_t_interp = interp1d(r_deg_interp, g_2d[:,i], kind='cubic')
            g_t = g_t_interp(self.WLdata['r_deg'])
            diff_ = self.WLdata['shear'] - g_t
            cov_miscenter_interp = RectBivariateSpline(r_deg_interp, r_deg_interp, cov_miscenter_)
            cov_miscenter_ = cov_miscenter_interp(self.WLdata['r_deg'], self.WLdata['r_deg'])
            full_cov_ = cov_miscenter_ + np.diag(self.WLdata['shearerr'])
            cho_f = cho_factor(full_cov_)
            pOfMass[i] = np.dot(diff_, cho_solve(cho_f, diff_))

        return pOfMass



    ########################################


    def like_Megacam(self, rho_c_z, Dl, delta_c, r_s):
        """Return array P(Megacam data|Mwl). Note that this is not normalized
        wrt the mArr for a good reason: In general, the mArr will not cover the
        full pOfMass range, and it varies as a function of SZ parameters.
        However, pOfMass is a product of normalized distributions, and so its
        normalization is constant throughout parameter space."""
        # Dimensionless radial distance [Radius][Mass]
        x = self.WLdata['r_deg'][:,None] * Dl * np.pi/180 / r_s[None,:]
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = 1.6624541593797974e+18/Dl/self.beta_avg

        # gamma_t, kappa, g_t [Radius][Mass]
        gamma_2d = get_Delta_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        kappa_2d = get_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        g_2d = gamma_2d/(1-kappa_2d) * (1 + kappa_2d*(self.beta2_avg/self.beta_avg**2-1))

        likelihood = norm.pdf(g_2d, self.WLdata['shear'][:,None], self.WLdata['shearerr'][:,None])

        pOfMass = np.prod(likelihood, axis=0)

        return pOfMass




    ########################################


    def like_HST(self, rho_c_z, Dl, delta_c, r_s):
        """Return array P(HST data|Mwl). Note that this is not normalized wrt
        the mArr for a good reason: In general, the mArr will not cover the full
        pOfMass range, and it varies as a function of SZ parameters. However,
        pOfMass is a product of normalized distributions, and so its
        normalization is constant throughout parameter space."""
        # Dimensionless radial distance [Radius][Mass]
        x = self.WLdata['r_deg'][:,None] * Dl * np.pi/180 / r_s[None,:]

        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2] [Radius]
        rangeR = range(len(self.WLdata['r_deg']))
        betaR = np.array([self.beta_avg[self.WLdata['magbinids'][i]] for i in rangeR])
        beta2R = np.array([self.beta2_avg[self.WLdata['magbinids'][i]] for i in rangeR])
        Sigma_c = 1.6624541593797974e+18/Dl/betaR

        # gamma_t and kappa [Radius][Mass]
        gamma_2d = get_Delta_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c[:,None]
        kappa_2d = get_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c[:,None]

        # [Radius][Mass]
        mu0_2d = 1/((1-kappa_2d)**2 - gamma_2d**2)
        kappaFake = (mu0_2d-1)/2

        # Magnification correction [Radius][Mass]
        mykappa = kappaFake * 0.3/betaR[:,None]

        magcorr = [np.interp(mykappa[i], self.WLdata['magcorr'][self.WLdata['magbinids'][i]][0], self.WLdata['magcorr'][self.WLdata['magbinids'][i]][1]) for i in rangeR]

        # Beta correction [Radius][Mass]
        betaratio = beta2R/betaR**2
        betaCorr = (1 + kappa_2d*(betaratio[:,None]-1))

        # Reduced shear g_t [Radius][Mass]
        g_2d = np.array(magcorr) * gamma_2d/(1-kappa_2d) * betaCorr

        # Only consider 500<r/kpc/1500 in reference cosmology
        cosmoRef = {'Omega_m':.3, 'Omega_l':.7, 'h':.7, 'w0':-1., 'wa':0}
        DlRef = cosmo.dA(self.zcluster, cosmoRef)
        rPhysRef = self.WLdata['r_deg'] * DlRef * np.pi/180 /cosmoRef['h']
        rInclude = np.where((rPhysRef>.5)&(rPhysRef<1.5))[0]

        likelihood = norm.pdf(g_2d[rInclude,:], self.WLdata['shear'][rInclude,None], self.WLdata['shearerr'][rInclude,None])

        pOfMass = np.prod(likelihood, axis=0)

        return pOfMass




    ########################################
    # dA [Mpc/h]
    def get_dAs(self, cosmology):
        """Precompute angular diameter distances for an array of redshifts."""
        zs = np.logspace(-1,np.log10(5),100)
        dA = np.array([cosmo.dA(z, cosmology) for z in zs])
        self.dAs = {'lnz':np.log(zs), 'lndA':np.log(dA)}


    ########################################
    def get_beta(self, cosmology):
        """Compute <beta> and <beta^2> from distribution of redshift galaxies."""
        ##### Only consider redshift bins behind the cluster
        betaArr = np.zeros(len(self.WLdata['redshifts']))
        bgIdx = np.where(self.WLdata['redshifts']>self.zcluster)[0]

        ##### Calculate beta(z_source)
        # Set up interpolation
        z_arr = np.linspace(np.amin(self.WLdata['redshifts'][bgIdx]), np.amax(self.WLdata['redshifts'][bgIdx]), 64)
        dA_ls = np.array([cosmo.dA_two_z(self.zcluster, z, cosmology) for z in z_arr])
        dA_ls_interp = interp1d(z_arr, dA_ls, kind='cubic')
        # beta = dA_ls / dA_l
        betaArr[bgIdx] = dA_ls_interp(self.WLdata['redshifts'][bgIdx])
        betaArr[bgIdx]/= np.exp(np.interp(np.log(self.WLdata['redshifts'][bgIdx]), self.dAs['lnz'], self.dAs['lndA']))

        ##### Weight beta(z) with N(z) distribution to get <beta> and <beta^2>
        if self.WLdata['datatype']=='Megacam':
            self.beta_avg = np.sum(self.WLdata['Nz']*betaArr)/self.WLdata['Ntot']
            self.beta2_avg = np.sum(self.WLdata['Nz']*betaArr**2)/self.WLdata['Ntot']
        elif self.WLdata['datatype']=='DES':
            self.beta_avg = np.mean(betaArr)
            self.beta2_avg = np.mean(betaArr**2)
        else:
            self.beta_avg, self.beta2_avg = {}, {}
            for i in self.WLdata['pzs'].keys():
                self.beta_avg[i] = np.sum(self.WLdata['pzs'][i]*betaArr)/self.WLdata['Ntot'][i]
                self.beta2_avg[i] = np.sum(self.WLdata['pzs'][i]*betaArr**2)/self.WLdata['Ntot'][i]


################################################################################
# NFW tools by Joerg Dietrich (https://github.com/joergdietrich/NFW)

def arcsec(z):
    """Compute the inverse sec of the complex number z."""
    val1 = 1j / z
    val2 = sm.sqrt(1 - 1/z**2)
    val = 1j * np.log(val2 + val1)
    return .5 * np.pi + val

def get_Delta_Sigma(x, r_s, rho_c_z, delta_c):
    """Return Delta Sigma"""
    fac = 2 * r_s * rho_c_z * delta_c
    val1 = 1 / (1 - x**2)
    num = ((3 * x**2) - 2) * arcsec(x)
    div = x**2 * (sm.sqrt(x**2 - 1))**3
    val2 = (num / div).real
    val3 = 2 * np.log(x / 2) / x**2
    return fac * (val1+val2+val3)

def get_Sigma(x, r_s, rho_c_z, delta_c):
    """Return Sigma_NFW"""
    fac = 2 * r_s * rho_c_z * delta_c
    val1 = 1 / (x**2 - 1)
    val2 = (arcsec(x) / (sm.sqrt(x**2 - 1))**3).real
    return fac * (val1-val2)



################################################################################
def readdata(catalog, HSTfile, MegacamFile, DESfile):
    """Read and load weak-lensing data into `WLdata` field in `catalog` if
    the corresponding path-variables lead to valid files on disk."""
    # Empty WL data field
    catalog['WLdata'] = [None for i in range(len(catalog['SPT_ID']))]

    ##### Check for HST data
    if HSTfile != 'None':
        with h5py.File(HSTfile, 'r') as f:
            for i,name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype':'HST', 'center':f[name].attrs['center'],
                        'r_deg':f[name]['shear_profile'][0], 'shear':f[name]['shear_profile'][1], 'shearerr':f[name]['shear_profile'][2], 'magbinids':f[name]['shear_profile'][3],
                        'redshifts':f[name]['redshifts'],
                        'pzs':{}, 'magcorr':{}, 'Ntot':{}}
                    for key in f[name]['magbindata'].keys():
                        catalog['WLdata'][i]['pzs'][key] = np.sum(f[name]['magbindata'][key]['pzs'], axis=0)
                        catalog['WLdata'][i]['Ntot'][key] = np.sum(catalog['WLdata'][i]['pzs'][key])
                        catalog['WLdata'][i]['magcorr'][key] = f[name]['magbindata'][key]['magnificationcorr']

    ##### Megacam data
    if MegacamFile != 'None':
        with h5py.File(MegacamFile, 'r') as f:
            for i,name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype':'Megacam',
                        'r_deg':f[name]['shear_profile'][0], 'shear':f[name]['shear_profile'][1], 'shearerr':f[name]['shear_profile'][2],
                        'redshifts':f[name]['Nz'][0], 'Nz':f[name]['Nz'][1], 'Ntot':np.sum(f[name]['Nz'][1]),}

    ##### Check for DES data
    if DESfile != 'None':
        with h5py.File(DESfile, 'r') as f:
            for i,name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype':'DES',
                        'r_deg':f[name]['shear_profile'][0], 'shear':f[name]['shear_profile'][1], 'shearerr':f[name]['shear_profile'][2],
                        'redshifts':f[name]['Nz'][:],}
