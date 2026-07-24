"""
pre_cmb.py

Computes the CMB compressed-distance-prior observables R and l_A, for
use with a Gaussian likelihood on (R, l_A, omega_b) - the standard
"CMB distance priors" compression (Bond, Efstathiou & Tegmark 1997;
Wang & Mukherjee 2007; various Planck-2018-chain reprocessing papers).

GEO/GROWTH SPLIT:
  - z_star, r_s(z_star)  : early-universe physics (recombination sound
                            horizon), computed from the physical/growth
                            sector densities (ombh2, omch2, omnuh2).
                            NOT split - these are real pre-recombination
                            quantities, same reasoning as rs_zdrag in
                            pre_bao.py.
  - D_M(z_star)           : late-time projection distance. THIS is the
                            geo-split quantity - computed with
                            omega_m_geo in the matter term of E(z).
  - omega_b = ombh2       : unchanged; read directly from
                            cosmological_parameters by the likelihood
                            module, not touched here.

IMPORTANT PHYSICS NOTE: unlike the SN/BAO distance modules, this one
integrates E(z) all the way out to z_star ~ 1090, where radiation is
NOT negligible - Omega_r*(1+z)^4 is already ~20-30% of Omega_m*(1+z)^3
by z~1090, since matter-radiation equality sits around z_eq~3400 for a
standard cosmology. cosmo.py's Ez has no radiation term and is only
valid in the SN/BAO z<~1.3 regime it was built for. This module
therefore does its OWN E(z) integration with an added radiation term,
rather than reusing cosmo.dA/cosmo.Ez. Omega_r is NOT a free/split
parameter - it is fixed entirely by TCMB and N_eff via the standard
photon+neutrino energy-density relations.

Assumes flat geometry (omega_k=0), matching this pipeline's
consistency-file default and cosmo.py's own assumption.

Writes to a custom 'cmb_distance_priors' section:
    z_star, r_s_star (Mpc), R, l_A
"""
import numpy as np
import scipy.integrate
from cosmosis.datablock import names, option_section

cosmo_section = names.cosmological_parameters
cmb_priors = "cmb_distance_priors"

C_KM_S = 299792.458  # km/s


def setup(options):
    return {}


def _omega_r(h0, tcmb, neff):
    """Fixed radiation density (photons + neutrinos) - not a free or
    split parameter, depends only on TCMB and N_eff."""
    omega_gamma_h2 = 2.469e-5 * (tcmb / 2.725) ** 4
    omega_nu_h2 = 0.2271 * neff * omega_gamma_h2
    omega_r_h2 = omega_gamma_h2 + omega_nu_h2
    return omega_r_h2 / h0 ** 2


def _z_star_hu_sugiyama(ombh2, omch2, omnuh2):
    """Hu & Sugiyama (1996) fitting formula for the photon-decoupling
    redshift. Uses the physical/growth-sector densities (NOT split)."""
    om_h2 = ombh2 + omch2 + omnuh2
    g1 = 0.0783 * ombh2 ** -0.238 / (1.0 + 39.5 * ombh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * ombh2 ** 1.81)
    z_star = 1048.0 * (1.0 + 0.00124 * ombh2 ** -0.738) * (1.0 + g1 * om_h2 ** g2)
    return z_star


def _sound_horizon_eh98(ombh2, omch2, omnuh2, tcmb, z_target):
    """Eisenstein & Hu (1998) closed-form sound horizon integral,
    evaluated out to an arbitrary target redshift (z_drag for BAO, or
    z_star here for the CMB). Uses the physical/growth-sector densities
    (NOT split) - same formula structure as pre_bao.py's version."""
    om_h2 = ombh2 + omch2 + omnuh2
    theta27 = tcmb / 2.7

    z_eq = 2.5e4 * om_h2 * theta27 ** -4
    k_eq = 7.46e-2 * om_h2 * theta27 ** -2  # Mpc^-1

    def baryon_photon_ratio(z):
        return 31.5 * ombh2 * theta27 ** -4 * (z / 1.0e3) ** -1

    R_target = baryon_photon_ratio(z_target)
    R_eq = baryon_photon_ratio(z_eq)

    s = (2.0 / (3.0 * k_eq)) * np.sqrt(6.0 / R_eq) * np.log(
        (np.sqrt(1.0 + R_target) + np.sqrt(R_target + R_eq)) / (1.0 + np.sqrt(R_eq))
    )
    return s


def execute(block, config):
    # --- geometric sector inputs ---
    omega_m_geo = block[cosmo_section, "omega_m_geo"]
    omega_lambda = block.get_double(cosmo_section, "omega_lambda", 1.0 - omega_m_geo)
    w0 = block.get_double(cosmo_section, "w", -1.0)
    wa = block.get_double(cosmo_section, "wa", 0.0)
    h0 = block[cosmo_section, "h0"]  # H0 / 100
    H0_phys = h0 * 100.0             # km/s/Mpc

    # --- growth/physical sector inputs ---
    ombh2 = block[cosmo_section, "ombh2"]
    omch2 = block[cosmo_section, "omch2"]
    omnuh2 = block.get_double(cosmo_section, "omnuh2", 0.0)
    tcmb = block.get_double(cosmo_section, "TCMB", 2.7255)
    neff = block.get_double(cosmo_section, "nnu", 3.046)

    # --- z_star and r_s(z_star): growth/physical sector, not split ---
    z_star = _z_star_hu_sugiyama(ombh2, omch2, omnuh2)
    r_s_star = _sound_horizon_eh98(ombh2, omch2, omnuh2, tcmb, z_star)

    # --- D_M(z_star): geometric sector (omega_m_geo), WITH radiation ---
    omega_r = _omega_r(h0, tcmb, neff)

    def E(z):
        de = (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / (1.0 + z))
        return np.sqrt(omega_r * (1.0 + z) ** 4
                        + omega_m_geo * (1.0 + z) ** 3
                        + omega_lambda * de)

    D_H = C_KM_S / H0_phys  # Mpc
    integral, _ = scipy.integrate.quad(lambda z: 1.0 / E(z), 0.0, z_star)
    D_M = D_H * integral    # physical Mpc, flat geometry

    # --- distance priors ---
    R = np.sqrt(omega_m_geo) * H0_phys / C_KM_S * D_M
    l_A = np.pi * D_M / r_s_star

    block[cmb_priors, "z_star"] = z_star
    block[cmb_priors, "r_s_star"] = r_s_star
    block[cmb_priors, "R"] = R
    block[cmb_priors, "l_A"] = l_A

    return 0


def cleanup(config):
    pass
