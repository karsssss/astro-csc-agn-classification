"""Load, merge, clean, and label the multiwavelength CSC catalog.

See CLAUDE.md ("Data conventions") for the sentinel-masking rule and the
gold/silver/unlabeled label taxonomy this module implements.
"""

import numpy as np
import pandas as pd


def load_catalog(path):
    """Read the main multiwavelength CSV and normalize the `name` index."""
    df = pd.read_csv(path, low_memory=False)
    df["name"] = df["name"].astype("string").str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.set_index("name")
    df = df.rename({"IR_agn": "ir_agn"}, axis=1)
    return df


def merge_csc_spectral_fits(df, path):
    """Cross-match CSC power-law spectral fit parameters (e.g. `powlaw_gamma`) on `name`."""
    spec = pd.read_csv(path)
    spec["name"] = (
        spec["name"].astype("string").str.strip().str.replace("2CXO ", "2CXO_", regex=False)
    )
    return df.merge(spec, on="name", how="left", validate="many_to_one")


def merge_desi(df, path):
    """Cross-match DESI spectroscopic redshifts/labels, OR-ing DESI's IR-AGN flag and
    redshift presence into the existing `ir_agn` column."""
    desi_cols = [
        "name", "DESI_TARGETID", "DESI_Z", "DESI_ZERR", "DESI_ZWARN",
        "DESI_SPECTYPE", "DESI_SUBTYPE", "combo_Z", "z_source", "high_quality_z",
        "combo_SPECTYPE", "spectype_source", "ir_agn",
    ]
    df_desi = pd.read_csv(path)
    df_desi["name"] = df_desi["name"].astype("string").str.strip()
    desi_small = df_desi[desi_cols].copy().rename({"ir_agn": "ir_agn_desi"}, axis=1)

    df = df.merge(desi_small, on="name", how="left", validate="many_to_one")
    df["ir_agn"] = (
        df["ir_agn"].fillna(False).astype(bool)
        | df["ir_agn_desi"].fillna(False).astype(bool)
        | df["combo_Z"].notna().astype(bool)
    )
    return df


def clean_types(df):
    """Coerce known boolean columns and blank strings to NaN across object columns."""
    df = df.copy()
    bool_cols = ["conf_flag", "extent_flag", "xray_agn", "optical_agn", "ir_agn"]
    df[bool_cols] = df[bool_cols].fillna(False).astype(bool)

    object_cols = df.select_dtypes(include="object").columns
    df[object_cols] = df[object_cols].astype("string").apply(lambda col: col.str.strip())
    df[object_cols] = df[object_cols].replace("", np.nan)
    return df


def mask_magnitude_sentinels(df, zero_point_jy):
    """Apply the project-wide sentinel rule (mag <= 0 -> NaN) to every raw magnitude
    column in `zero_point_jy`. Must run before any color/ratio/flux-conversion step
    that reads these columns -- see CLAUDE.md "Missing-value sentinels"."""
    df = df.copy()
    for col in zero_point_jy:
        df[col] = df[col].where(df[col] > 0)
    return df


def to_bool_safe(s):
    """Coerce a column that may be bool, or a string like 'True'/'1'/'yes', to bool."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype("string").str.strip().str.lower().isin(["true", "t", "1", "yes", "y"])


def mateos_wedge_mask(w2_w3, w1_w2):
    """WISE color-color AGN selection wedge (Mateos et al. 2012 piecewise-linear boundary)."""
    return (
        (w1_w2 <= 0.315 * w2_w3 + 0.796)
        & (w1_w2 >= 0.315 * w2_w3 - 0.222)
        & (w1_w2 >= -3.172 * w2_w3 + 7.264)
    )


def build_labels(df):
    """Build the gold/silver/unlabeled label taxonomy.

    Adds `agn_positive`, `agn_negative`, `label_conflict`, `silver_score`,
    `silver_agn`, and `label_3class` (-1 unlabeled, 0 confirmed non-AGN,
    1 silver AGN, 2 confirmed AGN). Priority is given to positive AGN labels
    over `non_agn` when both are set -- this is load-bearing for downstream
    model training, see CLAUDE.md.

    Requires merge_csc_spectral_fits and merge_desi to have already run
    (uses `powlaw_gamma`, `combo_Z`, `high_quality_z`), and mask_magnitude_sentinels
    to have already run on WISE/PS1/Legacy magnitude columns.
    """
    df = df.copy()
    agn_cols = ["xray_agn", "optical_agn", "ir_agn"]
    neg_col = "non_agn"
    for col in agn_cols + [neg_col]:
        df[col] = to_bool_safe(df[col])

    df["agn_positive"] = df[agn_cols].any(axis=1)
    df["agn_negative"] = df[neg_col]
    df["label_conflict"] = df["agn_positive"] & df["agn_negative"]
    df["agn_negative"] = df["agn_negative"] & ~df["agn_positive"]  # priority to positive labels

    wise_w1_w2 = df["WISE_w1mpro"] - df["WISE_w2mpro"]
    wise_w2_w3 = df["WISE_w2mpro"] - df["WISE_w3mpro"]
    mateos_wedge = mateos_wedge_mask(wise_w2_w3, wise_w1_w2)

    df["powlaw_gamma"] = pd.to_numeric(df["powlaw_gamma"], errors="coerce")
    has_csc_spectral_fit = df["powlaw_gamma"] > 0

    log_flux_aper_b = np.log10(df["flux_aper_b"].where(df["flux_aper_b"] > 0))
    xray_opt_r_relation_legacy = log_flux_aper_b + 0.4 * df["PS21P_r"]

    has_combo_z = pd.to_numeric(df["combo_Z"], errors="coerce").gt(0)
    if "high_quality_z" in df.columns:
        has_high_quality_combo_z = has_combo_z & to_bool_safe(df["high_quality_z"])
    else:
        has_high_quality_combo_z = has_combo_z

    silver_score = pd.Series(0.0, index=df.index)
    silver_score.loc[mateos_wedge.fillna(False)] += 0.35
    silver_score.loc[has_csc_spectral_fit] += 0.35
    q80_r = xray_opt_r_relation_legacy.quantile(0.80)
    silver_score.loc[xray_opt_r_relation_legacy >= q80_r] += 0.15
    silver_score.loc[has_high_quality_combo_z] += 0.15
    df["silver_score"] = silver_score.clip(0, 1)

    df["silver_agn"] = (
        (df["silver_score"] >= 0.01) & ~df["agn_positive"] & ~df["agn_negative"]
    )

    df["label_3class"] = np.select(
        [df["agn_positive"], df["agn_negative"], df["silver_agn"]],
        [2, 0, 1],
        default=-1,
    ).astype("int8")

    return df
