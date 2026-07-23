"""Magnitude -> flux conversion and cross-check utilities.

See CLAUDE.md, "Data conventions" section ("Missing-value sentinels",
"Photometric units and zero-points") for the project-wide rules this
module implements.
"""

import numpy as np
import pandas as pd

# Zero-point table (Jy) per magnitude column, keyed by photometric system.
# AB system (flat 3631 Jy by definition): Pan-STARRS DR2, DESI Legacy Survey
# Vega system (per-band zero-point): WISE (Jarrett et al. 2011), 2MASS (Cohen et al. 2003)
# Gaia G is its own Vega-like system; zero-point below is approximate (Gaia DR2/EDR3).
ZERO_POINT_JY = {
    # Pan-STARRS DR2 (AB)
    "PS21P_g": 3631.0,
    "PS21P_r": 3631.0,
    "PS21P_i": 3631.0,
    "PS21P_z": 3631.0,
    # DESI Legacy Survey (AB)
    "LEG21P_G": 3631.0,
    "LEG21P_R": 3631.0,
    "LEG21P_Z": 3631.0,
    # WISE (Vega; Jarrett et al. 2011)
    "WISE_w1mpro": 309.540,
    "WISE_w2mpro": 171.787,
    "WISE_w3mpro": 31.674,
    # 2MASS (Vega; Cohen et al. 2003)
    "2MASS21P_j_m": 1594.0,
    "2MASS21P_h_m": 1024.0,
    "2MASS21P_k_m": 666.7,
    # Gaia G (Vega-like; approximate DR2/EDR3 zero-point)
    "GAIA21P_g": 3228.75,
}

# Very approximate 5-sigma point-source depths (WISE: Wright et al. 2010; Pan-STARRS1:
# Chambers et al. 2016; DESI Legacy Survey: Dey et al. 2019, varies by region/pass; 2MASS:
# 10-sigma PSC limits; Gaia: rough DR3 G depth), in magnitude -- used only for an
# order-of-magnitude plausibility check, not independently re-derived.
DEPTH_MAG_APPROX = {
    "PS21P_g": 23.3, "PS21P_r": 23.2, "PS21P_i": 23.1, "PS21P_z": 22.3,
    "LEG21P_G": 24.7, "LEG21P_R": 23.9, "LEG21P_Z": 23.0,
    "WISE_w1mpro": 16.6, "WISE_w2mpro": 15.7, "WISE_w3mpro": 11.4,
    "2MASS21P_j_m": 15.8, "2MASS21P_h_m": 15.1, "2MASS21P_k_m": 14.3,
    "GAIA21P_g": 21.0,
}


def mag_to_flux_jy(mag, zero_point_jy):
    """Convert a magnitude Series to flux density in Jy.

    Every survey in ZERO_POINT_JY encodes non-detections with its own magic
    sentinel (WISE/PS21P use -999, LEG21P uses -9); a real magnitude is
    always > 0, so masking on that instead of chasing per-survey sentinels
    keeps this consistent across all bands (see CLAUDE.md, "Missing-value
    sentinels").
    """
    mag_valid = mag.where(mag > 0)
    return zero_point_jy * 10 ** (-0.4 * mag_valid)


def apply_mag_to_flux_conversion(df, zero_points=ZERO_POINT_JY, depth_mag_approx=DEPTH_MAG_APPROX, verbose=True):
    """Apply mag_to_flux_jy to every column in `zero_points`, in place on a copy.

    For each column `col` this adds `flux_jy_{col}` (linear flux, Jy) and
    `log_flux_{col}` (log10 flux, Jy -- the scale that matches the existing
    X-ray `log_flux_aper_{b,h,s}` columns). Columns already given as fluxes
    (`SDSS_CALIBFLUX_*`, `log_flux_aper_*`) are not in `zero_points` and are
    left untouched.

    If `verbose`, prints one diagnostic line per band: how many sentinel
    values were masked, the median magnitude, the median flux in a
    human-scale unit (mJy for WISE, uJy otherwise), and an order-of-magnitude
    plausibility check against `depth_mag_approx`.

    Returns the same `df` (mutated) for convenient chaining: `df = apply_mag_to_flux_conversion(df)`.
    """
    for col, zp in zero_points.items():
        mag = df[col]
        n_sentinel = int((mag <= 0).sum())

        flux_jy = mag_to_flux_jy(mag, zp)
        log_flux = np.log10(flux_jy.where(flux_jy > 0))

        df[f"flux_jy_{col}"] = flux_jy
        df[f"log_flux_{col}"] = log_flux

        if not verbose:
            continue

        median_mag = mag[mag > 0].median()
        median_flux_jy = flux_jy.median()

        if col.startswith("WISE_"):
            unit_val, unit_name = median_flux_jy * 1e3, "mJy"
        else:
            unit_val, unit_name = median_flux_jy * 1e6, "uJy"

        depth = depth_mag_approx.get(col) if depth_mag_approx else None
        if depth is not None:
            plaus = "within typical depth" if median_mag <= depth else "FAINTER than typical depth (check)"
        else:
            plaus = "no depth reference"

        print(f"{col:16s} | masked={n_sentinel:6d} | median_mag={median_mag:6.2f} | "
              f"median_flux={unit_val:8.3f} {unit_name} | {plaus}")

    return df


def cross_check_mag_zero_point(df, mag_col, ref_flux_col, ref_flux_unit="nanomaggie", label=None):
    """Regression-test pattern for a mag->flux zero-point.

    Converts an independent, already-calibrated flux column for the same
    physical sources into a magnitude in mag_col's system, then compares
    directly. A near-zero median offset confirms mag_col's zero-point/
    system; a large systematic offset flags a wrong zero-point, wrong
    system, or a mislabeled column -- this is exactly what caught (and then
    ruled out) an apparent PS21P_r discrepancy during development.

    Returns the per-row magnitude difference (mag_col - ref_mag) Series.
    """
    mask = (df[mag_col] > 0) & df[ref_flux_col].notna() & (df[ref_flux_col] > 0)
    sub = df.loc[mask, [mag_col, ref_flux_col]]

    if ref_flux_unit == "nanomaggie":
        ref_mag = 22.5 - 2.5 * np.log10(sub[ref_flux_col])
    else:
        raise ValueError(f"Unsupported ref_flux_unit: {ref_flux_unit!r}")

    diff = sub[mag_col] - ref_mag
    label = label or f"{mag_col} vs {ref_flux_col}"
    print(f"{label}: n={len(sub)}, median diff={diff.median():+.3f} mag, "
          f"IQR=({diff.quantile(0.25):+.3f}, {diff.quantile(0.75):+.3f})")
    return diff
