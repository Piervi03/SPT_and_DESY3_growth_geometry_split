"""
derive_cmb_distance_priors.py

OFFLINE, ONE-TIME script - run this yourself, locally, after downloading a
public Planck chain (e.g. from the Planck Legacy Archive,
https://pla.esac.esa.int, the "base_Alens_plikHM_TTTEEE_lowl_lowE" chain
to match the A_lens-marginalized approach discussed earlier). This is NOT
a CosmoSIS module and is NOT run as part of your pipeline - it produces a
small data file (mean vector + covariance matrix) that the CosmoSIS
likelihood module (cmb_distance_prior_like_5d.py) reads at runtime.

What it does, per MCMC sample in the chain:
  1. Reads the already-sampled parameters: ombh2, omch2, H0 (or h0), ns,
     and As (or logA, converted).
  2. Computes z_star (Hu & Sugiyama 1996 fit) and r_s(z_star)
     (Eisenstein & Hu 1998 closed form) - same formulas as pre_cmb.py.
  3. Computes D_M(z_star) via direct numerical integration of the
     radiation-inclusive E(z), same physics as pre_cmb.py's approach
     (but here using the chain's own single Omega_m - this is REAL data
     being summarized, not a geo/growth-split theory prediction, so
     there is no split to apply here).
  4. Computes R and l_A from those.
Then takes the WEIGHTED mean and covariance of the 5D vector
    x = (R, l_A, ombh2, ns, 1e9*As)
across all samples (MCMC chains carry a multiplicity/weight per row -
using an unweighted mean/cov would bias the result) and writes them to
a text file for the likelihood module to load.

====================================================================
YOU MUST CHECK / EDIT the CONFIG block below against your actual
downloaded chain's .paramnames file before running this. Column names
and the presence/absence of a native 'zstar' derived parameter vary
between chain releases - I cannot verify these against your specific
download since I don't have access to it.
====================================================================
"""
import numpy as np
import scipy.integrate
import getdist

# ------------------------- CONFIG - CHECK THIS -------------------------
CHAIN_ROOT = "./base_Alens_plikHM_TTTEEE_lowl_lowE"  # no file extension

# Column names as they appear in your chain's .paramnames file.
# Check that file and adjust these strings if they differ.
COL_OMBH2 = "omegabh2"
COL_OMCH2 = "omegach2"
COL_H0 = "H0"          # Hubble constant in km/s/Mpc (NOT h)
COL_NS = "ns"
COL_LOGA = "logA"      # ln(1e10 * A_s) 
COL_AS = None          # stores A_s directly - set exactly one of these two
COL_ZSTAR = "zstar"    
                       
COL_OMNUH2 =  None     # set to a column name if present; else uses
                       # OMNUH2_FIXED below (typical single massive nu
                       # minimal-mass default)
OMNUH2_FIXED =0.0 #0.00064
TCMB = 2.7255
NEFF = 3.046           # fixed for standard base_Alens chain (w=-1, flat)

THIN = 1               # set >1 to subsample every Nth row for speed
Z_GRID_MAX = 1200.0    # comfortably above any realistic z_star
Z_GRID_N = 2000

OUTPUT_FILE = "cmb_distance_priors_5d.txt"
# -------------------------------------------------------------------

C_KM_S = 299792.458  # km/s


def _z_star_hu_sugiyama(ombh2, omch2, omnuh2):
    om_h2 = ombh2 + omch2 + omnuh2
    g1 = 0.0783 * ombh2 ** -0.238 / (1.0 + 39.5 * ombh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * ombh2 ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * ombh2 ** -0.738) * (1.0 + g1 * om_h2 ** g2)


def _sound_horizon_eh98(ombh2, omch2, omnuh2, tcmb, z_target):
    om_h2 = ombh2 + omch2 + omnuh2
    theta27 = tcmb / 2.7
    z_eq = 2.5e4 * om_h2 * theta27 ** -4
    k_eq = 7.46e-2 * om_h2 * theta27 ** -2

    def baryon_photon_ratio(z):
        return 31.5 * ombh2 * theta27 ** -4 * (z / 1.0e3) ** -1

    R_target = baryon_photon_ratio(z_target)
    R_eq = baryon_photon_ratio(z_eq)
    return (2.0 / (3.0 * k_eq)) * np.sqrt(6.0 / R_eq) * np.log(
        (np.sqrt(1.0 + R_target) + np.sqrt(R_target + R_eq)) / (1.0 + np.sqrt(R_eq))
    )


def _omega_r(h0, tcmb, neff):
    omega_gamma_h2 = 2.469e-5 * (tcmb / 2.725) ** 4
    omega_nu_h2 = 0.2271 * neff * omega_gamma_h2
    return (omega_gamma_h2 + omega_nu_h2) / h0 ** 2


def _D_M_single(omega_m, omega_r, omega_lambda, H0_phys, z_star):
    """One numerical integral per sample - slower than a vectorized
    approach but simple, memory-safe, and easy to check for correctness.
    A chain of ~1e4-1e5 samples still finishes in well under a minute."""
    def E(z):
        return np.sqrt(omega_r * (1.0 + z) ** 4
                        + omega_m * (1.0 + z) ** 3
                        + omega_lambda)  # w=-1 fixed for base_Alens chain

    D_H = C_KM_S / H0_phys
    integral, _ = scipy.integrate.quad(lambda z: 1.0 / E(z), 0.0, z_star)
    return D_H * integral


def main():
    print("Loading chain:", CHAIN_ROOT)
    samples = getdist.loadMCSamples(CHAIN_ROOT)

    def get(name):
        return samples.samples[:, samples.paramNames.numberOfName(name)]

    ombh2 = get(COL_OMBH2)[::THIN]
    omch2 = get(COL_OMCH2)[::THIN]
    H0 = get(COL_H0)[::THIN]
    ns = get(COL_NS)[::THIN]
    weights = samples.weights[::THIN]

    if COL_AS is not None:
        As = get(COL_AS)[::THIN]
    else:
        logA = get(COL_LOGA)[::THIN]
        As = np.exp(logA) * 1.0e-10

    if COL_OMNUH2 is not None:
        omnuh2 = get(COL_OMNUH2)[::THIN]
    else:
        omnuh2 = np.full_like(ombh2, OMNUH2_FIXED)

    n = len(ombh2)
    print(f"Using {n} samples (thin={THIN})")

    if COL_ZSTAR is not None:
        z_star = get(COL_ZSTAR)[::THIN]
    else:
        print("No native zstar column configured - computing via "
              "Hu & Sugiyama fit instead.")
        z_star = _z_star_hu_sugiyama(ombh2, omch2, omnuh2)

    r_s_star = _sound_horizon_eh98(ombh2, omch2, omnuh2, TCMB, z_star)

    H0_phys = H0  # already km/s/Mpc
    h0 = H0 / 100.0
    omega_m = (ombh2 + omch2 + omnuh2) / h0 ** 2
    omega_r = _omega_r(h0, TCMB, NEFF)
    omega_lambda = 1.0 - omega_m - omega_r  # flat

    print("Integrating D_M(z_star) per sample - this is the slow step...")
    D_M = np.empty(n)
    for i in range(n):
        D_M[i] = _D_M_single(omega_m[i], omega_r[i], omega_lambda[i],
                              H0_phys[i], z_star[i])
        if i % 2000 == 0:
            print(f"  {i}/{n}")

    R = np.sqrt(omega_m) * H0_phys / C_KM_S * D_M
    l_A = np.pi * D_M / r_s_star

    x = np.column_stack([R, l_A, ombh2, ns, 1.0e9 * As])
    labels = ["R", "l_A", "ombh2", "ns", "1e9*As"]

    mean = np.average(x, axis=0, weights=weights)
    delta = x - mean
    # weighted covariance
    cov = (delta * weights[:, None]).T @ delta / weights.sum()

    print("\nMean vector:")
    for lbl, m in zip(labels, mean):
        print(f"  {lbl:8s} = {m:.6g}")
    print("\nCovariance matrix:")
    print(cov)

    with open(OUTPUT_FILE, "w") as f:
        f.write("# order: " + " ".join(labels) + "\n")
        f.write("# mean vector\n")
        f.write(" ".join(f"{v:.10e}" for v in mean) + "\n")
        f.write("# covariance matrix\n")
        for row in cov:
            f.write(" ".join(f"{v:.10e}" for v in row) + "\n")

    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
