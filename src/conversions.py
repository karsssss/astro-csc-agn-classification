"""Magnitude -> flux conversion and the v_final Reduced Case feature matrix builder.

See CLAUDE.md ("Photometric units and zero-points") for the zero-point
sources and the physical reasoning behind the ratio/color choices below, and
results/SHAP_SUMMARY.md for why soft-/hard-band X-ray ratios are excluded.
"""

import re

import numpy as np
import pandas as pd

# Zero-point table (Jy), keyed by raw magnitude column. AB system (3631 Jy
# universal): Pan-STARRS DR2, DESI Legacy Survey. Vega system (per-band):
# WISE (Jarrett et al. 2011), 2MASS (Cohen et al. 2003). Gaia G is its own
# Vega-like system (approximate DR2/EDR3 zero-point).
ZERO_POINT_JY = {
    "PS21P_g": 3631.0, "PS21P_r": 3631.0, "PS21P_i": 3631.0, "PS21P_z": 3631.0,
    "LEG21P_G": 3631.0, "LEG21P_R": 3631.0, "LEG21P_Z": 3631.0,
    "WISE_w1mpro": 309.540, "WISE_w2mpro": 171.787, "WISE_w3mpro": 31.674,
    "2MASS21P_j_m": 1594.0, "2MASS21P_h_m": 1024.0, "2MASS21P_k_m": 666.7,
    "GAIA21P_g": 3228.75,
}

# The definitive 25-column "Reduced Case" set the notebook's own cell 29
# dropna's on -- the ONLY starting columns v_final's feature matrix is built
# from.
XRAY_FLUX_COLS = [
    "significance",
    "flux_aper_b", "flux_aper_lolim_b", "flux_aper_hilim_b",
    "flux_aper_h", "flux_aper_lolim_h", "flux_aper_hilim_h",
    "flux_aper_s", "flux_aper_lolim_s", "flux_aper_hilim_s",
    "flux_aper_m", "flux_aper_lolim_m", "flux_aper_hilim_m",
]
WISE_RAW_COLS = ["WISE_w1mpro", "WISE_w1snr", "WISE_w2mpro", "WISE_w2snr", "WISE_w3mpro", "WISE_w3snr"]
HARDNESS_COLS = [
    "hard_hs", "hard_hs_lolim", "hard_hs_hilim",
    "hard_hm", "hard_hm_lolim", "hard_hm_hilim",
    "hard_ms", "hard_ms_lolim", "hard_ms_hilim",
]
DEFINITIVE_REDUCED_CASE_COLS = XRAY_FLUX_COLS + ["combo_g", "combo_r", "combo_z_mag"] + WISE_RAW_COLS + [
    "hard_hs", "hard_hs_lolim", "hard_hs_hilim",
]

_RAW_FLUX_LEAK_PATTERN = re.compile(r"flux_(aper|combo|wise)_[a-z0-9]+$")


def mag_to_flux_jy(mag, zero_point_jy):
    """Convert a magnitude Series to flux density in Jy.

    Masks non-positive sentinel values (WISE/PS21P use -999, LEG21P uses -9;
    a real magnitude is always > 0) to NaN before converting -- see CLAUDE.md
    "Missing-value sentinels".
    """
    mag_valid = mag.where(mag > 0)
    return zero_point_jy * 10 ** (-0.4 * mag_valid)


def apply_mag_to_flux(df, column_zero_points, flux_col_template="flux_{col}"):
    """Add a `flux_<col>` column (Jy) for every `col: zero_point_jy` pair.

    Narrower than the pre-restructure `apply_mag_to_flux_conversion`:
    explicit column specification required (no implicit "convert everything
    in ZERO_POINT_JY" default), and no per-band diagnostic printing or
    automatic log10 companion column -- see conversions.py's
    cross_check_mag_zero_point for verification instead.
    """
    df = df.copy()
    for col, zp in column_zero_points.items():
        df[flux_col_template.format(col=col)] = mag_to_flux_jy(df[col], zp)
    return df


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


def cross_check_mag_zero_point(df, mag_col, ref_flux_col, ref_flux_unit="nanomaggie", label=None):
    """Regression-test pattern for a mag->flux zero-point.

    Methodological infrastructure from CLAUDE.md's "Verification discipline
    for physical-units work" -- not called by build_feature_matrix at
    runtime; this is a verification utility for when new bands/zero-points
    are added, not a production code path.

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


def combine_first_column(df, primary_col, fallback_col, out_col):
    """Return a copy of df with out_col = primary_col, falling back to fallback_col
    where primary is NaN (e.g. PS1 with DESI Legacy Survey fallback)."""
    df = df.copy()
    df[out_col] = df[primary_col].combine_first(df[fallback_col])
    return df


def drop_raw_flux_columns(df):
    """Drop any raw single-band flux/magnitude-derived flux column
    (`flux_aper_*`, `flux_combo_*`, `flux_wise_*`) -- only their ratios/colors
    should survive into a feature matrix."""
    leaked = [c for c in df.columns if _RAW_FLUX_LEAK_PATTERN.match(c)]
    return df.drop(columns=leaked)


def _log_ratio(numer, denom):
    ratio = numer / denom
    return np.log10(ratio.where(ratio > 0))


def build_feature_matrix(df):
    """Build the v_final Reduced Case feature matrix: 25 raw columns -> mag->flux
    conversion -> broad/medium-band X-ray ratios + WISE/optical colors + hardness
    ratios + SNR/significance. Zero raw flux/magnitude columns in the output
    (asserted). Returns (df_sample, X) where df_sample retains all original
    columns (e.g. for label-based reporting) and X is the numeric feature matrix.

    Soft- and hard-band X-ray ratios are deliberately excluded: they require
    flux_aper_s/flux_aper_h to be strictly positive (not just non-null), which
    reintroduces a brightness/detection selection effect (see
    results/SHAP_SUMMARY.md). hard_hs/hard_hm/hard_ms carry hard/soft spectral
    shape without that effect.

    Requires df's raw magnitude columns (PS21P_*, LEG21P_*, WISE_*mpro) to have
    already been sentinel-masked (see preprocessing.mask_magnitude_sentinels)
    and combo_g/combo_r/combo_z_mag to already exist (PS1-with-Legacy-Survey
    fallback via combine_first_column).
    """
    missing = [c for c in DEFINITIVE_REDUCED_CASE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"build_feature_matrix requires these columns first: {missing}")

    df_sample = df.dropna(subset=DEFINITIVE_REDUCED_CASE_COLS).copy()

    df_sample = apply_mag_to_flux(df_sample, {
        "combo_g": ZERO_POINT_JY["PS21P_g"],
        "combo_r": ZERO_POINT_JY["PS21P_r"],
        "combo_z_mag": ZERO_POINT_JY["PS21P_z"],
        "WISE_w1mpro": ZERO_POINT_JY["WISE_w1mpro"],
        "WISE_w2mpro": ZERO_POINT_JY["WISE_w2mpro"],
        "WISE_w3mpro": ZERO_POINT_JY["WISE_w3mpro"],
    }, flux_col_template="flux_{col}")
    # apply_mag_to_flux names outputs flux_combo_g/flux_combo_r/flux_combo_z_mag/
    # flux_WISE_w1mpro/etc; rename the WISE ones to the shorter names ratios use below.
    df_sample = df_sample.rename(columns={
        "flux_WISE_w1mpro": "flux_wise_w1",
        "flux_WISE_w2mpro": "flux_wise_w2",
        "flux_WISE_w3mpro": "flux_wise_w3",
        "flux_combo_z_mag": "flux_combo_z",
    })

    df_sample["xray_b_over_r"] = _log_ratio(df_sample["flux_aper_b"], df_sample["flux_combo_r"])
    df_sample["xray_b_over_w1"] = _log_ratio(df_sample["flux_aper_b"], df_sample["flux_wise_w1"])
    df_sample["xray_b_over_w2"] = _log_ratio(df_sample["flux_aper_b"], df_sample["flux_wise_w2"])
    df_sample["xray_m_over_r"] = _log_ratio(df_sample["flux_aper_m"], df_sample["flux_combo_r"])
    df_sample["optical_r_over_w1"] = _log_ratio(df_sample["flux_combo_r"], df_sample["flux_wise_w1"])

    df_sample["color_g_over_r"] = _log_ratio(df_sample["flux_combo_g"], df_sample["flux_combo_r"])
    df_sample["color_r_over_z"] = _log_ratio(df_sample["flux_combo_r"], df_sample["flux_combo_z"])
    df_sample["color_w1_over_w2"] = _log_ratio(df_sample["flux_wise_w1"], df_sample["flux_wise_w2"])
    df_sample["color_w2_over_w3"] = _log_ratio(df_sample["flux_wise_w2"], df_sample["flux_wise_w3"])

    final_cols = [
        "significance", "WISE_w1snr", "WISE_w2snr", "WISE_w3snr",
        "hard_hs", "hard_hs_lolim", "hard_hs_hilim",
        "hard_hm", "hard_hm_lolim", "hard_hm_hilim",
        "hard_ms", "hard_ms_lolim", "hard_ms_hilim",
        "xray_b_over_r", "xray_b_over_w1", "xray_b_over_w2", "xray_m_over_r",
        "optical_r_over_w1",
        "color_g_over_r", "color_r_over_z", "color_w1_over_w2", "color_w2_over_w3",
    ]
    missing_final = [c for c in final_cols if c not in df_sample.columns]
    if missing_final:
        raise KeyError(f"final_cols references columns not present: {missing_final}")

    df_sample = df_sample.dropna(subset=final_cols)
    df_sample = drop_raw_flux_columns(df_sample)

    X = df_sample[final_cols].copy()
    leaked = [c for c in X.columns if _RAW_FLUX_LEAK_PATTERN.match(c)]
    assert not leaked, f"raw flux/magnitude columns leaked into feature matrix: {leaked}"

    return df_sample, X
