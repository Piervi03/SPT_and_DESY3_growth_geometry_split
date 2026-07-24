import numpy as np
from math import sqrt as msqrt
import camb
from camb import model
from cosmosis.datablock import option_section

#REview all of this to see if it's correct
def setup(options):
    """Return the redshift array and grid settings needed to build the
    CAMB linear power spectrum on each call to execute()."""
    compute_sigma8 = options.get_bool(option_section, 'compute_sigma8', False)
    z_min_max = options.get_double_array_1d(option_section, 'z_min_max')
    N_z = options.get_int(option_section, 'N_z')
    z_arr = np.linspace(z_min_max[0], z_min_max[1], N_z)

    # k sampling for the linear P(k), in h/Mpc -- exposed as options so you
    # can tune resolution/range without touching the code.
    minkh = options.get_double(option_section, 'minkh', 1e-4)
    maxkh = options.get_double(option_section, 'maxkh', 5.0)
    nk = options.get_int(option_section, 'nk', 400)
    kmax = options.get_double(option_section, 'kmax', 10.0)  # passed to CAMB internally, should be >= maxkh

    k_arr = np.logspace(np.log10(minkh), np.log10(maxkh), nk)

    return compute_sigma8, z_arr, k_arr, kmax


def _camb_cold_pk_interpolator(params, zmax, kmax):
    """
    Build a CAMB linear "cold" (CDM+baryon, i.e. no massive-neutrino
    contribution) matter power spectrum interpolator P(z, k_h) for a given
    parameter dictionary, mirroring baccoemu's cold=True option.

    params keys (same names as in the original bacco dictionaries):
        omega_matter, omega_baryon, neutrino_mass, hubble, ns, w0, wa, A_s
    """
    h = params['hubble']
    H0 = 100. * h
    ombh2 = params['omega_baryon'] * h ** 2
    mnu = params['neutrino_mass']

    # Standard approximation for the present-day massive-neutrino density,
    # Omega_nu h^2 = sum(m_nu [eV]) / 93.14, valid for the usual Neff = 3.046
    # normalization. omega_matter is assumed to be the TOTAL matter density
    # parameter (CDM + baryons + massive neutrinos), consistent with the
    # standard cosmological convention -- adjust the line below if your
    # sampler instead defines omega_matter as CDM+baryons only.
    omch2 = params['omega_matter'] * h ** 2 - ombh2

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu)
    # 'ppf' dark energy model handles w0/wa combinations that cross w = -1,
    # which the simpler 'fluid' model cannot.
    pars.set_dark_energy(w=params['w0'], wa=params['wa'], dark_energy_model='ppf')
    pars.InitPower.set_params(As=params['A_s'], ns=params['ns'])

    PK = camb.get_matter_power_interpolator(
        pars,
        nonlinear=False,
        hubble_units=True,
        k_hunit=True,
        kmax=kmax,
        zmax=zmax + 0.5,
        var1=model.Transfer_nonu,
        var2=model.Transfer_nonu,
    )
    return PK


def execute(block, config):
    """Read cosmological parameters, run CAMB for the geometry and growth
    cosmologies, and write the recombined power spectrum to the block."""
    compute_sigma8, z_arr, k, kmax = config

    params_growth = {'omega_matter': block.get_double('cosmological_parameters', 'Omega_m_growth'),
                      'omega_baryon': block.get_double('cosmological_parameters', 'Omega_b'),
                      'neutrino_mass': block.get_double('cosmological_parameters', 'mnu'),
                      'hubble': block.get_double('cosmological_parameters', 'h0'),
                      'ns': block.get_double('cosmological_parameters', 'n_s'),
                      'w0': block.get_double('cosmological_parameters', 'w'),
                      'wa': block.get_double('cosmological_parameters', 'wa'),
                      'A_s': block.get_double('cosmological_parameters', 'A_s'),
                      }
    params_geo = {'omega_matter': block.get_double('cosmological_parameters', 'Omega_m_geo'),
                  'omega_baryon': block.get_double('cosmological_parameters', 'Omega_b'),
                  'neutrino_mass': block.get_double('cosmological_parameters', 'mnu'),
                  'hubble': block.get_double('cosmological_parameters', 'h0'),
                  'ns': block.get_double('cosmological_parameters', 'n_s'),
                  'w0': block.get_double('cosmological_parameters', 'w'),
                  'wa': block.get_double('cosmological_parameters', 'wa'),
                  'A_s': block.get_double('cosmological_parameters', 'A_s'),
                  }
    z_i = block.get_double('cosmological_parameters', 'z_i')

    zmax_needed = max(z_i, z_arr.max())

    # One CAMB run per cosmology gives a spline interpolator we can then
    # evaluate as many times as needed -- much cheaper than one run per z.
    PK_geo = _camb_cold_pk_interpolator(params_geo, zmax_needed, kmax)
    PK_growth = _camb_cold_pk_interpolator(params_growth, zmax_needed, kmax)

    Pk_geo_zi = PK_geo.P(z_i, k)
    Pk_growth_zi = PK_growth.P(z_i, k)
    Pk_growth_z = np.array([PK_growth.P(z, k) for z in z_arr])

    # Recombined power spectrum: the geometry cosmology sets the amplitude
    # at z_i, the growth cosmology sets the shape and its z-evolution.
    ratio_zi = Pk_geo_zi / Pk_growth_zi
    Pk = ratio_zi[None, :] * Pk_growth_z

    block.put_grid('cdm_baryon_power_lin', 'z', z_arr, 'k_h', k, 'p_k', Pk)

    # Compute sigma_8 for cold dark matter here, otherwise no split is possible
    if compute_sigma8:
        Pk_growth_0 = PK_growth.P(0., k)
        Pk_ = ratio_zi * Pk_growth_0

        kR = 8. * k
        window = 3. * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)
        integrand_sigma2 = Pk_ * window ** 2 * k ** 3
        sigma8_squ = .5 / np.pi ** 2 * np.trapezoid(integrand_sigma2, np.log(k))
        block.put_double('cosmological_parameters', 'sigma_8', msqrt(sigma8_squ))

    return 0


def cleanup(config):
    pass