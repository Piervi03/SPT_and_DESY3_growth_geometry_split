import numpy as np
from numpy.lib import scimath as sm
from scipy.interpolate import make_interp_spline
from multiprocessing import Pool
import h5py

import cosmo
import Mconversion_concentration
import miscentering

# Limits for stack
z_names = ['zlo', 'zmid', 'zhi']
xi_names = ['xilo', 'xihi']
z_lims = [.25, .4, .6, .95]
xi_lims = [4.25, 5.5, 1e10]

# Mass [Msun/h]
grid_lgM_min = 12.5
grid_lgM_max = 16.
grid_lgM_N = 32
lnM_arr_default = np.log(10.)*np.linspace(grid_lgM_min, grid_lgM_max, grid_lgM_N)


def unwrap_self_one_cluster(arg):
    """Wrapper function needed to make multiprocessing work within a class"""
    return SPTlensing.one_cluster(*arg)


class SPTlensing:
    """Read lensing data and compute ln P(shear | M)."""

    def __init__(self, catalog, **kwargs):
        # Set up more stuff
        self.save_shear_profiles = kwargs.pop('save_shear_profiles', False)
        self.NPROC = kwargs.pop('NPROC', 0)
        # Read lensing data
        self.Euclidfile = kwargs.pop('Euclidfile', 'None')
        self.DESfile = kwargs.pop('DESfile')
        self.HSTfile = kwargs.pop('HSTfile')
        self.MegacamFile = kwargs.pop('MegacamFile')
        readdata(catalog,
                 self.HSTfile,
                 self.MegacamFile,
                 self.DESfile,
                 self.Euclidfile,
                 self.save_shear_profiles)
        # Redshift range of clusters with WL data
        self.WL_idx = [dat is not None for dat in catalog['WLdata']]
        self.z_cl_min = np.amin(catalog['REDSHIFT'][self.WL_idx])
        self.z_cl_max = np.amax(catalog['REDSHIFT'][self.WL_idx])
        # Mass conversion for HST and Megacam (legacy)
        if (self.HSTfile != 'None') or (self.MegacamFile != 'None'):
            self.mcType = kwargs.pop('mcType')
            self.Delta_crit = kwargs.pop('Delta_crit')
        # DES-specific stuff
        if self.DESfile != 'None':
            # source redshifts
            with h5py.File(self.DESfile, 'r') as f:
                self.DES = {'SOM_Z_MID': f['config/SOM_Z_MID'][:],
                            'SOM_BINs': f['config/SOM_BINs'][:][1:, :]}
            # Read boost chain
            DESboostfile = kwargs.pop('DESboostfile')
            with open(DESboostfile, 'r') as f:
                tmp = f.readline().split()[1:]
            dat = np.mean(np.loadtxt(DESboostfile), axis=0)
            self.DES['boost_dict'] = {'z_arr': kwargs.pop('DESboost_z_arr')}
            for n, name in enumerate(tmp):
                self.DES['boost_dict'][name] = dat[n]
            # Initialize miscentering
            DESmiscenterfile = kwargs.pop('DESmiscenterfile')
            DEScentertype = kwargs.pop('DEScentertype')
            with open(DESmiscenterfile, 'r') as f:
                tmp = f.readline().split()[1:]
            dat = np.mean(np.loadtxt(DESmiscenterfile), axis=0)
            miscenter_dict = {}
            for n, name in enumerate(tmp):
                miscenter_dict[name] = dat[n]
            miscenter_dict['SPT'] = {'kind': DEScentertype, 'kappa_SPT': miscenter_dict['kappa_SPT']}
            miscenter_dict['MCMF'] = {'kind': DEScentertype}
            for glob, this in zip(['alpha_SZ_0', 'alpha_SZ_z', 'alpha_SZ_lam',
                                   'SZ_comp0_0', 'SZ_comp0_z', 'SZ_comp0_lam',
                                   'SZ_comp1_0', 'SZ_comp1_z', 'SZ_comp1_lam'],
                                  ['alpha_0', 'alpha_z', 'alpha_lam',
                                   'comp0_0', 'comp0_z', 'comp0_lam',
                                   'comp1_0', 'comp1_z', 'comp1_lam']):
                miscenter_dict['SPT'][this] = miscenter_dict[glob]
            for glob, this in zip(['alpha_opt_0', 'alpha_opt_z', 'alpha_opt_lam',
                                   'opt_comp0_0', 'opt_comp0_z', 'opt_comp0_lam',
                                   'opt_comp1_0', 'opt_comp1_z', 'opt_comp1_lam'],
                                  ['alpha_0', 'alpha_z', 'alpha_lam',
                                   'comp0_0', 'comp0_z', 'comp0_lam',
                                   'comp1_0', 'comp1_z', 'comp1_lam']):
                miscenter_dict['MCMF'][this] = miscenter_dict[glob]
            self.DES['miscenterer'] = miscentering.MisCentering(miscenter_dict[DEScentertype])
        # Euclid-specific stuff
        if self.Euclidfile != 'None':
            # Source redshift distribution
            with h5py.File(self.Euclidfile, 'r') as f:
                self.Euclid = {'z_s': f['z_s'][:],
                               'tomo_dist': f['tomo_dist'][:]}
            # Read boost chain
            Euclidboost_z_arr = kwargs.pop('Euclidboost_z_arr')
            self.Euclid['boost_dict'] = {'z_arr': Euclidboost_z_arr}
            Euclidboostfile = kwargs.pop('Euclidboostfile')
            with open(Euclidboostfile, 'r') as f:
                tmp = f.readline().split()[1:]
            dat = np.mean(np.loadtxt(Euclidboostfile), axis=0)
            for n, name in enumerate(tmp):
                self.Euclid['boost_dict'][name] = dat[n]
            # Initialize miscentering
            Euclidcentertype = kwargs.pop('Euclidcentertype')
            Euclidmiscenterfile = kwargs.pop('Euclidmiscenterfile')
            with open(Euclidmiscenterfile, 'r') as f:
                tmp = f.readline().split()[1:]
            dat = np.mean(np.loadtxt(Euclidmiscenterfile), axis=0)
            miscenter_dict = {}
            for n, name in enumerate(tmp):
                miscenter_dict[name] = dat[n]
            miscenter_dict['SPT'] = {'kind': Euclidcentertype, 'kappa_SPT': miscenter_dict['kappa_SPT']}
            miscenter_dict['MCMF'] = {'kind': Euclidcentertype}
            for glob, this in zip(['alpha_SZ_0', 'alpha_SZ_z', 'alpha_SZ_lam',
                                   'SZ_comp0_0', 'SZ_comp0_z', 'SZ_comp0_lam',
                                   'SZ_comp1_0', 'SZ_comp1_z', 'SZ_comp1_lam'],
                                  ['alpha_0', 'alpha_z', 'alpha_lam',
                                   'comp0_0', 'comp0_z', 'comp0_lam',
                                   'comp1_0', 'comp1_z', 'comp1_lam']):
                miscenter_dict['SPT'][this] = miscenter_dict[glob]
            for glob, this in zip(['alpha_opt_0', 'alpha_opt_z', 'alpha_opt_lam',
                                   'opt_comp0_0', 'opt_comp0_z', 'opt_comp0_lam',
                                   'opt_comp1_0', 'opt_comp1_z', 'opt_comp1_lam'],
                                  ['alpha_0', 'alpha_z', 'alpha_lam',
                                   'comp0_0', 'comp0_z', 'comp0_lam',
                                   'comp1_0', 'comp1_z', 'comp1_lam']):
                miscenter_dict['MCMF'][this] = miscenter_dict[glob]
            self.Euclid['miscenterer'] = miscentering.MisCentering(miscenter_dict[Euclidcentertype])

    def lnlike_all(self, catalog, cosmology, scaling):
        """Return ln p(data|M_arr) for all clusters with WL data."""
        # t = []
        # t.append(time.time())
        self.cosmology = cosmology
        self.scaling = scaling
        if (self.HSTfile != 'None') or (self.MegacamFile != 'None'):
            if self.Delta_crit == 200.:
                self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology,
                                                                               setup_interp=False)
            else:
                self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology,
                                                                               setup_interp=True,
                                                                               interp_massdef=self.Delta_crit)
        if self.DESfile != 'None':
            self.MCrel_DES = Mconversion_concentration.ConcentrationConversion(3.5)
        self.lnM_arr = lnM_arr_default
        self.M_arr = np.exp(self.lnM_arr)
        # Pre-compute angular diameter distances
        self.lndA_interp, self.dA_twoz_interp = cosmo.get_dAs(self.z_cl_min, self.z_cl_max, 5., cosmology)
        # t.append(time.time())

        if self.NPROC == 0:
            res = [self.one_cluster(catalog[i]) for i in self.WL_idx]
        else:
            with Pool(processes=self.NPROC) as pool:
                argin = zip([self]*len(self.WL_idx), catalog[self.WL_idx])
                res = pool.map(unwrap_self_one_cluster, argin)

        catalog['lnp_Mwl'] = [None]*len(catalog)
        catalog['lnp_Mwl'][self.WL_idx] = [r[0] for r in res]
        if self.save_shear_profiles:
            catalog['model_shear_profile'] = [None]*len(catalog)
            catalog['model_shear_profile'][self.WL_idx] = [r[1] for r in res]
        # t.append(time.time())
        # print("lensing done", t[-1]-t[0])

        return 0

    def setup_one_cluster_mode(self, cosmology):
        """Function name says it all. Call this function before calling
        `one_cluster` directly. Not relevant if you are calling `lnlike_all`."""
        self.cosmology = cosmology
        if (self.HSTfile != 'None') or (self.MegacamFile != 'None'):
            if self.Delta_crit == 200.:
                self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, cosmology,
                                                                               setup_interp=False)
            else:
                self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, cosmology,
                                                                               setup_interp=True,
                                                                               interp_massdef=self.Delta_crit)
        if self.DESfile != 'None':
            self.MCrel_DES = Mconversion_concentration.ConcentrationConversion(3.5)
        # Pre-compute angular diameter distances
        self.lndA_interp, self.dA_twoz_interp = cosmo.get_dAs(self.z_cl_min, self.z_cl_max, 5., cosmology)

    def one_cluster(self, data, M_arr=None):
        """Process the cluster given by `data`. Return ln-likelihood of shear
        profile, additionally return model shear profiles depending on
        `save_shear_profiles`."""
        # Set up
        # t = [time.time()]
        self.cat_cl = data
        if M_arr is None:
            M_arr = self.M_arr
        #  D_ls / D_s, needed for lensing efficiency
        self.get_beta(self.cosmology)
        # t.append(time.time())
        # Likelihood
        if self.cat_cl['WLdata']['datatype'] in ['DES', 'Euclid']:
            if self.cat_cl['WLdata']['datatype'] == 'DES':
                self.modeldict = self.DES
            else:
                self.modeldict = self.Euclid
            res = self.DESEuclid_cluster(M_arr)
        elif self.cat_cl['WLdata']['datatype'] == 'Megacam':
            res = self.Megacam_cluster(M_arr)
        elif self.cat_cl['WLdata']['datatype'] == 'HST':
            res = self.HST_cluster(M_arr)
        # t.append(time.time())
        # print('done', t[-1]-t[0], np.diff(t))
        return res

    def DESEuclid_cluster(self, M_arr):
        """Return array lnP(DES data|Mwl) and optionally the model shear profiles."""
        # Model
        Dl = np.exp(self.lndA_interp(np.log(self.cat_cl['REDSHIFT'])))
        r_Mpch = self.cat_cl['WLdata']['r_arcmin'] * Dl * np.pi/60/180
        invSigma_c = Dl*self.beta_avg/cosmo.c2_4piG
        R_mis = self.modeldict['miscenterer'].get_mean_Rmis(self.cat_cl, self.cosmology)
        c200c = self.MCrel_DES.calC200(M_arr, self.cat_cl['REDSHIFT']) * np.ones(len(M_arr))
        reduced_shear_bins = shear_model_flatNFW_clmemcont(M_arr, self.cat_cl['REDSHIFT'], c200c,
                                                           r_Mpch, invSigma_c, R_mis, self.cosmology)
        reduced_shear = (np.sum(self.cat_cl['WLdata']['tomo_rescale'][:, None, None] * self.cat_cl['WLdata']['tomo_weights'].T[:, None, :] * reduced_shear_bins, axis=0)
                         / np.sum(self.cat_cl['WLdata']['tomo_weights'], axis=1))
        # Cluster member contamination
        A = boost_get_A('Gausssmooth',
                        self.cat_cl['REDSHIFT'], self.cat_cl['richness'], r_Mpch, R_mis,
                        **self.modeldict['boost_dict'])
        reduced_shear_cont = 1/(1+A) * reduced_shear
        # Likelihood
        lnP_DES_Mwl = -.5*np.sum(((reduced_shear_cont-self.cat_cl['WLdata']['shear'])/self.cat_cl['WLdata']['shear_err'])**2, axis=1)
        if self.save_shear_profiles:
            return lnP_DES_Mwl, reduced_shear_cont
        else:
            return lnP_DES_Mwl, None

    def shear_model_Megacam(self, mass):
        """Return Megacam shear model given mass."""
        Dl = np.exp(self.lndA_interp(np.log(self.cat_cl['REDSHIFT'])))
        # NFW halo stuff
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.cat_cl['REDSHIFT'], self.cosmology)**2  # [h^2 Msun/Mpc^3]
        if self.Delta_crit == 200.:
            M200c = mass
        else:
            M200c = np.exp(self.MCrel.lnM_to_lnM200(self.cat_cl['REDSHIFT'], np.log(mass)))[0]
        r200c = (3*M200c/4/np.pi/200/rho_c_z)**(1/3)
        c200c = self.MCrel.calC200(M200c, self.cat_cl['REDSHIFT'])
        delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
        r_s = r200c/c200c
        # Dimensionless radial distance [Radius][Mass]
        x = self.cat_cl['WLdata']['r_deg'][:, None] * Dl * np.pi/180 / r_s[None, :]
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
        Sigma_c = cosmo.c2_4piG/Dl/self.beta_avg
        # gamma_t, kappa, g_t [Radius][Mass]
        gamma_2d = get_DeltaSigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        kappa_2d = get_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c
        g_2d = gamma_2d/(1-kappa_2d) * (1 + kappa_2d*(self.beta2_avg/self.beta_avg**2-1))
        return g_2d

    def Megacam_cluster(self, M_arr):
        """Return array lnP(Megacam data|Mwl) and optionally the model shear profiles."""
        # Model
        g_2d = self.shear_model_Megacam(M_arr)
        # Likelihood
        lnpOfMass = -.5*np.sum(((g_2d-self.cat_cl['WLdata']['shear'][:, None])/self.cat_cl['WLdata']['shearerr'][:, None])**2, axis=0)
        if self.save_shear_profiles:
            return lnpOfMass, g_2d
        else:
            return lnpOfMass, None

    def shear_model_HST(self, mass):
        """Return HST shear model given mass."""
        Dl = np.exp(self.lndA_interp(np.log(self.cat_cl['REDSHIFT'])))
        # NFW halo stuff
        rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.cat_cl['REDSHIFT'], self.cosmology)**2  # [h^2 Msun/Mpc^3]
        if self.Delta_crit == 200.:
            M200c = mass
        else:
            M200c = np.exp(self.MCrel.lnM_to_lnM200(self.cat_cl['REDSHIFT'], np.log(mass)))[0]
        r200c = (3*M200c/4/np.pi/200/rho_c_z)**(1/3)
        c200c = self.MCrel.calC200(M200c, self.cat_cl['REDSHIFT'])
        delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
        r_s = r200c/c200c
        # Dimensionless radial distance [Radius][Mass]
        x = self.cat_cl['WLdata']['r_deg'][:, None] * Dl * np.pi/180 / r_s[None, :]
        # Sigma_crit, with c^2/4piG [h Msun/Mpc^2] [Radius]
        rangeR = range(len(self.cat_cl['WLdata']['r_deg']))
        betaR = np.array([self.beta_avg[self.cat_cl['WLdata']['magbinids'][i]] for i in rangeR])
        beta2R = np.array([self.beta2_avg[self.cat_cl['WLdata']['magbinids'][i]] for i in rangeR])
        Sigma_c = cosmo.c2_4piG/Dl/betaR
        # gamma_t and kappa [Radius][Mass]
        gamma_2d = get_DeltaSigma(x, r_s, rho_c_z, delta_c) / Sigma_c[:, None]
        kappa_2d = get_Sigma(x, r_s, rho_c_z, delta_c) / Sigma_c[:, None]
        # [Radius][Mass]
        mu0_2d = 1/((1-kappa_2d)**2 - gamma_2d**2)
        kappaFake = (mu0_2d-1)/2
        # Magnification correction [Radius][Mass]
        mykappa = kappaFake * 0.3/betaR[:, None]
        magcorr = [np.interp(mykappa[i], self.cat_cl['WLdata']['magcorr'][self.cat_cl['WLdata']['magbinids'][i]][0], self.cat_cl['WLdata']['magcorr'][self.cat_cl['WLdata']['magbinids'][i]][1])
                   for i in rangeR]
        # Beta correction [Radius][Mass]
        betaratio = beta2R/betaR**2
        betaCorr = (1 + kappa_2d*(betaratio[:, None]-1))
        # Reduced shear g_t [Radius][Mass]
        g_2d = np.array(magcorr) * gamma_2d/(1-kappa_2d) * betaCorr
        return g_2d

    def HST_cluster(self, M_arr):
        """Return array lnP(HST data|Mwl) and optionally the model shear profiles."""
        # Only consider 500<r/kpc/1500 in reference cosmology
        cosmoRef = {'Omega_m': .3, 'Omega_l': .7, 'h': .7, 'w0': -1., 'wa': 0}
        DlRef = cosmo.dA(self.cat_cl['REDSHIFT'], cosmoRef)
        rPhysRef = self.cat_cl['WLdata']['r_deg'] * DlRef * np.pi / 180 / cosmoRef['h']
        rInclude = (rPhysRef > .5) & (rPhysRef < 1.5)
        # Model
        g_2d = self.shear_model_HST(M_arr)
        # Likelihood
        lnpOfMass = -.5 * np.sum(((g_2d[rInclude, :]-self.cat_cl['WLdata']['shear'][rInclude, None])
                                  / self.cat_cl['WLdata']['shearerr'][rInclude, None])**2, axis=0)
        if self.save_shear_profiles:
            return lnpOfMass, g_2d
        else:
            return lnpOfMass, None

    def get_beta(self, cosmology):
        if self.cat_cl['WLdata']['datatype'] == 'DES':
            self.get_beta_DES(cosmology)
        elif self.cat_cl['WLdata']['datatype'] == 'Euclid':
            self.get_beta_Euclid(cosmology)
        else:
            self.get_beta_HST_Megacam(cosmology)
        return 0

    def get_beta_Euclid(self, cosmology):
        """Return mean(beta) for each Euclid tomo bin."""
        beta = np.zeros(len(self.Euclid['z_s']))
        bgIdx = self.Euclid['z_s'] > self.cat_cl['REDSHIFT']
        beta[bgIdx] = (self.dA_twoz_interp(np.log(self.cat_cl['REDSHIFT']), np.log(self.Euclid['z_s'][bgIdx]))[0]
                       / np.exp(self.lndA_interp(np.log(self.Euclid['z_s'][bgIdx]))))
        self.beta_avg = np.sum(self.Euclid['tomo_dist']*beta[None, :], axis=1)/np.sum(self.Euclid['tomo_dist'], axis=1)
        return 0

    def get_beta_DES(self, cosmology):
        """Return mean(beta) for each DES tomo bin."""
        beta = np.zeros(len(self.DES['SOM_Z_MID']))
        bgIdx = self.DES['SOM_Z_MID'] > self.cat_cl['REDSHIFT']
        beta[bgIdx] = (self.dA_twoz_interp(np.log(self.cat_cl['REDSHIFT']), np.log(self.DES['SOM_Z_MID'][bgIdx]))[0]
                       / np.exp(self.lndA_interp(np.log(self.DES['SOM_Z_MID'][bgIdx]))))
        self.beta_avg = np.sum(self.DES['SOM_BINs']*beta[None, :], axis=1)/np.sum(self.DES['SOM_BINs'], axis=1)
        return 0

    def get_beta_HST_Megacam(self, cosmology):
        """Compute <beta> and <beta^2> from distribution of redshift galaxies."""
        # Redshift bins behind the cluster
        betaArr = np.zeros(len(self.cat_cl['WLdata']['redshifts']))
        bgIdx = self.cat_cl['WLdata']['redshifts'] > self.cat_cl['REDSHIFT']
        # beta = dA_ls / dA_l
        betaArr[bgIdx] = (self.dA_twoz_interp(np.log(self.cat_cl['REDSHIFT']), np.log(self.cat_cl['WLdata']['redshifts'][bgIdx]))[0]
                          / np.exp(self.lndA_interp(np.log(self.cat_cl['WLdata']['redshifts'][bgIdx]))))
        # Weight beta(z) with N(z) distribution to get <beta> and <beta^2>
        if self.cat_cl['WLdata']['datatype'] == 'Megacam':
            self.beta_avg = np.average(betaArr, weights=self.cat_cl['WLdata']['Nz'])
            self.beta2_avg = np.average(betaArr**2, weights=self.cat_cl['WLdata']['Nz'])
        else:
            self.beta_avg, self.beta2_avg = {}, {}
            for i in self.cat_cl['WLdata']['pzs'].keys():
                self.beta_avg[i] = np.sum(self.cat_cl['WLdata']['pzs'][i]*betaArr)/self.cat_cl['WLdata']['Ntot'][i]
                self.beta2_avg[i] = np.sum(self.cat_cl['WLdata']['pzs'][i]*betaArr**2)/self.cat_cl['WLdata']['Ntot'][i]
        return 0


####################################################################
# NFW tools by Joerg Dietrich (https://github.com/joergdietrich/NFW)

def arcsec(z):
    """Compute the inverse sec of the complex number z."""
    val1 = 1j / z
    val2 = sm.sqrt(1 - 1/z**2)
    val = 1j * np.log(val2 + val1)
    return .5 * np.pi + val


def get_DeltaSigma(x, r_s, rho_c_z, delta_c):
    """Return Delta Sigma = Sigma - Sigma_mean"""
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


def get_Sigma_mis(r_Mpch, r_s, rho_c_z, delta_c, R_mis):
    """Return Sigma(r) for an NFW profile but where Sigma is constant within
    `R_mis`. `r_Mpch` and `r_s` can be arrays (if `r_s` is array then `delta_c`
    should have same size), `R_mis` must be scalar."""
    Sigma = get_Sigma(r_Mpch/r_s, r_s, rho_c_z, delta_c)
    Sigma_NFW_at_Rmis = get_Sigma(R_mis/r_s, r_s, rho_c_z, delta_c)
    ones = np.ones(Sigma.shape)
    const_idx = r_Mpch*ones < R_mis
    Sigma[const_idx] = (Sigma_NFW_at_Rmis*ones)[const_idx]
    return Sigma


def get_DeltaSigma_mis(r_Mpch, r_s, rho_c_z, delta_c, R_mis):
    """Return DeltaSigma(r) for an NFW profile but where Sigma is constant
    within `R_mis`. `r_Mpch` and `r_s` can be arrays (if `r_s` is array then
    `delta_c` should have same size), `R_mis` must be scalar."""
    # NFW Sigma and DeltaSigma
    x = r_Mpch/r_s
    DeltaSigma_NFW = get_DeltaSigma(x, r_s, rho_c_z, delta_c)
    # NFW Sigma and DeltaSigma at Rmis
    x_Rmis = R_mis/r_s
    DeltaSigma_NFW_at_Rmis = get_DeltaSigma(x_Rmis, r_s, rho_c_z, delta_c)
    # Miscentered DeltaSigma
    DeltaSigma = DeltaSigma_NFW - (R_mis/r_Mpch)**2 * DeltaSigma_NFW_at_Rmis
    DeltaSigma[r_Mpch*np.ones(DeltaSigma.shape) < R_mis] = 0.
    return DeltaSigma


def boost_get_A(method, z, lam, r, Rmis, **kwargs):
    """Compute A(z, lambda, r) where fcl = A/(1+A)."""
    if method in ['z_bins', 'Gausssmooth', 'Gauss_step', 'spline']:
        z_arr = kwargs.pop('z_arr')
        if method in ['Gauss_step', 'Gausssmooth']:
            corr_len = kwargs.pop('corr_len')
            A_inf = kwargs.pop('A_inf')
    logc = kwargs.pop('logc')
    Blambda = kwargs.pop('Blambda')
    # Radial dependence normalized to 1 at 1Mpc/h [r]
    r_s = (lam/60)**(1/3) / 10**logc
    P_r = get_Sigma(r/r_s, r_s, 1, 1) / get_Sigma(1/r_s, r_s, 1, 1)
    # Simplified model for miscentered profile
    P_at_Rmis = get_Sigma(Rmis/r_s, r_s, 1, 1) / get_Sigma(1/r_s, r_s, 1, 1)
    P_r[r < Rmis] = P_at_Rmis
    # Redshift dependence
    if method == 'z_bins':
        z_idx = np.digitize(z, z_arr)-1
        A_z = np.exp(kwargs.pop('A_%d' % z_idx))
    elif method == 'Gausssmooth':
        amps = [kwargs.pop('A_%d' % i) for i in range(len(z_arr))]
        A_z = np.exp(A_inf + np.sum(amps * np.exp(-.5*(z-z_arr)**2/corr_len**2)))
    elif method == 'Gauss_step':
        z_step = kwargs.pop('z_step')
        z_step_idx = np.digitize(z, z_step)-1
        z_arr_idx = ((z_arr >= z_step[z_step_idx]) & (z_arr < z_step[z_step_idx+1])).nonzero()[0]
        amps = [kwargs['A_{}'.format(i)] for i in z_arr_idx]
        A_z = np.exp(A_inf + np.sum(amps * np.exp(-.5*(z-z_arr[z_arr_idx])**2/corr_len**2)))
    elif method == 'spline':
        z_step = kwargs.pop('z_step')
        z_step_idx = (z < z_step).nonzero()[0][0] - 1
        z_arr_idx = ((z_arr >= z_step[z_step_idx]) & (z_arr < z_step[z_step_idx+1])).nonzero()[0]
        amps = 10.**(np.array([kwargs['A_{}'.format(i)] for i in z_arr_idx]))
        A_z = make_interp_spline(z_arr[z_arr_idx], amps, k=2)(z)
    elif method == 'const':
        A_z = np.exp(kwargs.pop('lnA'))
    # Richness dependence
    A_lambda = (lam/60)**Blambda
    # Put it all together [r]
    A = A_z * A_lambda * P_r
    return A


def shear_model_flatNFW_clmemcont(mass, z, c200c, r_Mpch, invSigma_c, R_mis, cosmology):
    """Return shear model [tomobin][mass][radius] given `mass`. Sigma follows an NFW that is constant
    within `R_mis`."""
    # NFW halo stuff
    rho_c_z = cosmo.RHOCRIT * cosmo.Ez(z, cosmology)**2
    r200c = (3*mass/4/np.pi/200/rho_c_z)**(1/3)
    delta_c = 200/3 * c200c**3 / (np.log(1+c200c) - c200c/(1+c200c))
    r_s = r200c/c200c
    # NFW surface mass densities
    r_Mpch = np.atleast_2d(r_Mpch)
    Sigma_mis = get_Sigma_mis(r_Mpch[:, None, :], r_s[None, :, None], rho_c_z, delta_c[None, :, None], R_mis)
    DeltaSigma_mis = get_DeltaSigma_mis(r_Mpch[:, None, :], r_s[None, :, None], rho_c_z, delta_c[None, :, None], R_mis)
    # Reduced shear profile
    invSigma_c = np.atleast_1d(invSigma_c)
    reduced_shear = DeltaSigma_mis*invSigma_c[:, None, None] / (1 - Sigma_mis*invSigma_c[:, None, None])
    return reduced_shear


def get_invSigmac_DES(z, SOM_Z_MID, SOM_BINs, lndA_interp, dA_twoz_interp):
    """Return mean(invSigmac) for each DES tomo bin."""
    beta = np.zeros(len(SOM_Z_MID))
    bgIdx = SOM_Z_MID > z
    beta[bgIdx] = (dA_twoz_interp(np.log(z), np.log(SOM_Z_MID[bgIdx]))[0]
                   / np.exp(lndA_interp(np.log(SOM_Z_MID[bgIdx]))))
    beta_avg = np.sum(SOM_BINs*beta[None, :], axis=1) / np.sum(SOM_BINs, axis=1)
    invSigmac = np.exp(lndA_interp(np.log(z))) * beta_avg / cosmo.c2_4piG
    return invSigmac


######################################################################################
def readdata(catalog, HSTfile, MegacamFile, DESfile, Euclidfile, save_shear_profiles):
    """Read and load weak-lensing data into `WLdata` field in `catalog`. The order
    matters, as data will be overwritten."""
    # Empty WL data field
    catalog['WLdata'] = [None for i in range(len(catalog['SPT_ID']))]
    # DES
    if DESfile != 'None':
        with h5py.File(DESfile, 'r') as f:
            for i, name in enumerate(catalog['SPT_ID']):
                if name in f['clusters'].keys():
                    catalog['WLdata'][i] = {'datatype': 'DES',
                                            'r_arcmin': f['clusters'][name]['r_arcmin'][:],
                                            'shear': f['clusters'][name]['shear'][:],
                                            'shear_err': f['clusters'][name]['shear_err'][:],
                                            'tomo_weights': f['clusters'][name]['tomo_weights_R'][:][:, 1:],
                                            'tomo_rescale': f['clusters'][name]['tomo_rescale'][1:],
                                            }
                    if save_shear_profiles:
                        z_idx = np.digitize(catalog['REDSHIFT'][i], z_lims)-1
                        xi_idx = np.digitize(catalog['XI'][i], xi_lims)-1
                        catalog['WLdata'][i]['r_arcmin_stack'] = f['stack_%s%s' % (z_names[z_idx], xi_names[xi_idx])]['r_arcmin'][:]
    # Megacam
    if MegacamFile != 'None':
        with h5py.File(MegacamFile, 'r') as f:
            for i, name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype': 'Megacam',
                                            'r_deg': f[name]['shear_profile'][0],
                                            'shear': f[name]['shear_profile'][1],
                                            'shearerr': f[name]['shear_profile'][2],
                                            'redshifts': f[name]['Nz'][0],
                                            'Nz': f[name]['Nz'][1], 'Ntot': np.sum(f[name]['Nz'][1])}
    # Euclid
    if Euclidfile != 'None':
        with h5py.File(Euclidfile, 'r') as f:
            for i, name in enumerate(catalog['SPT_ID']):
                if name in f['clusters'].keys():
                    catalog['WLdata'][i] = {'datatype': 'Euclid',
                                            'r_arcmin': f['clusters'][name]['r_arcmin'][:],
                                            'shear': f['clusters'][name]['shear'][:],
                                            'shear_err': f['clusters'][name]['shear_err'][:],
                                            'tomo_weights': f['clusters'][name]['tomo_weights_R'][:],
                                            'tomo_rescale': f['clusters'][name]['tomo_rescale'][:],
                                            }
                    if save_shear_profiles:
                        z_idx = np.digitize(catalog['REDSHIFT'][i], z_lims)-1
                        xi_idx = np.digitize(catalog['XI'][i], xi_lims)-1
                        catalog['WLdata'][i]['r_arcmin_stack'] = f['stack_%s%s' % (z_names[z_idx], xi_names[xi_idx])]['r_arcmin'][:]
    # HST
    if HSTfile != 'None':
        with h5py.File(HSTfile, 'r') as f:
            for i, name in enumerate(catalog['SPT_ID']):
                if name in f.keys():
                    catalog['WLdata'][i] = {'datatype': 'HST',
                                            'center': f[name].attrs['center'],
                                            'r_deg': f[name]['shear_profile'][0, :],
                                            'shear': f[name]['shear_profile'][1, :],
                                            'shearerr': f[name]['shear_profile'][2, :],
                                            'magbinids': f[name]['magbinid'][:],
                                            'redshifts': f[name]['redshifts'][:],
                                            'pzs': {},
                                            'magcorr': {},
                                            'Ntot': {}}
                    for key in f[name]['magbindata'].keys():
                        dict_key = int(key)
                        catalog['WLdata'][i]['pzs'][dict_key] = f[name]['magbindata'][key]['pzs'][:]
                        catalog['WLdata'][i]['Ntot'][dict_key] = np.sum(catalog['WLdata'][i]['pzs'][dict_key])
                        catalog['WLdata'][i]['magcorr'][dict_key] = f[name]['magbindata'][key]['magnificationcorr'][:]
    return 0
