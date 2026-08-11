"""Load, merge, clean, and label the multiwavelength CSC catalog.

See CLAUDE.md ("Data conventions") for the sentinel-masking rule and the
gold/silver/unlabeled label taxonomy this module implements.
"""

import numpy as np
import pandas as pd
from astropy.table import Table


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


def merge_kim23_galaxies(df, path):
    """Cross-match the Kim et al. 2023 Chandra Galaxy Catalog (CGC) on `name`,
    adding external galaxy photometry/redshift/classification and a `galaxy` flag.

    Wraps Alec's original merge logic verbatim (column cut, rename map, outer
    join, per-column combine_first) -- only his hardcoded personal FITS path
    is swapped for `path` (use "data/raw/Kim23_CGC.fits"), `galaxy` is
    filled/cast to a proper bool after his merge (his version leaves it as
    True/NaN), and -999 sentinels in Kim23's own magnitude columns are masked
    to NaN after the merge (see caveat below -- his version did not do this).

    Outer join, but empirically a no-op on row count against the current
    170,196-row catalog snapshot: every one of Kim23_CGC.fits's 8,547 CSCIDs
    already matches an existing `name` here (0 new rows). Kept as
    `how="outer"` per Alec's original design in case a future catalog
    release includes CSCIDs not yet cross-matched.

    combine_first is one-directional -- it only fills NaNs in df's existing
    columns from Kim23's data, never overwrites an already-non-null value.
    Several kept/renamed columns collide with existing catalog columns and
    get backfilled this way: `ra`, `dec`, `significance`, `flux_aper_b`,
    `flux_aper_h`, `Rec_Z`, `2MASS21P_k_m`, `WISE_w1mpro`/`w1snr`/`w2mpro`/
    `w2snr`/`w3mpro`/`w3snr`, `SDSS_SPECOBJID`, `PS21P_objID`, `LEG21P_OBJID`,
    `2MASS21P_designation`, `SDSS_CLASS`, `SDSS_SUBCLASS`, `z_source`.
    `combo_r_mag`/`combo_g_mag`/`galaxy`/`Gal_z_err`/`WISEID`/`GALEXID`/`spec`
    are new columns, no collision -- note `combo_r_mag`/`combo_g_mag` are the
    Kim23 galaxy counterpart's own photometry and are NOT the same thing as
    this pipeline's own `combo_r`/`combo_g` (built by combine_first_column
    from PS21P_r/g + LEG21P_R/G for the X-ray source's optical counterpart --
    do not conflate the two).

    `flux_aper_b`/`flux_aper_h` are unit-converted from Kim23's FxB15/FxH15
    (x1e-15) and cross-checked against this catalog's own flux_aper_b on
    4,556 overlap rows: median ratio 1.0000024, i.e. the same units.

    Sentinel handling (fixed here, unlike Alec's original): Kim23's own raw
    g/W1/W2/W3 columns carry unmasked -999 values (267 / ~1027-1029 of 8,547
    rows respectively) -- per CLAUDE.md's project-wide "mag <= 0 -> NaN" rule
    (sentinels overflow mag_to_flux to inf), this function masks
    combo_g_mag/WISE_w1mpro/WISE_w2mpro/WISE_w3mpro to NaN wherever <= 0,
    applied after combine_first so no sentinel can reach a downstream
    mag_to_flux conversion.

    Label-taxonomy note: every current `agn_negative` row (6,037/6,037) and
    every current `label_conflict` row (2,510/2,510) is `galaxy=True` after
    this merge -- i.e. Kim23 (or an earlier version of it, already reflected
    in this catalog's pre-existing Rec_Z/WISE coverage) is the source of
    this catalog's confirmed-non-AGN labels. Relevant for any downstream
    AGN-vs-non-AGN work: `agn_negative` is not an independent signal from
    `galaxy`, it is (so far) a subset of it.
    """
    dw_gals = Table.read(path).to_pandas()
    dw_gals["flux_aper_b"] = dw_gals["FxB15"] * 10 ** (-15)
    dw_gals["flux_aper_h"] = dw_gals["FxH15"] * 10 ** (-15)

    str_cols = ["CSCID", "SDSSID", "PSID", "LEGSID", "WISEID", "2MASSID", "GALEXID",
                "z_code", "class_SDSS", "subclass_SDSS"]
    float_cols = ["ra", "dec", "z", "zErr", "r", "g", "k", "W1", "w1snr", "W2", "w2snr",
                  "W3", "w3snr", "significance", "flux_aper_b", "flux_aper_h"]
    # 1 if galaxy selected by SDSS, 0 if not
    int_cols = ["spec"]
    dw_gals = dw_gals[str_cols + float_cols + int_cols]

    for col in str_cols:
        dw_gals[col] = dw_gals[col].astype(str).str.strip()
    for col in float_cols:
        dw_gals[col] = dw_gals[col].astype(float)
    for col in int_cols:
        dw_gals[col] = dw_gals[col].astype(int)

    cols_to_rename = ["CSCID", "SDSSID", "PSID", "LEGSID", "2MASSID", "z_code", "class_SDSS",
                       "subclass_SDSS", "z", "zErr", "r", "g", "k", "W1", "w1snr", "W2",
                       "w2snr", "W3", "w3snr"]
    cols_renamed = ["name", "SDSS_SPECOBJID", "PS21P_objID", "LEG21P_OBJID",
                    "2MASS21P_designation", "z_source", "SDSS_CLASS", "SDSS_SUBCLASS",
                    "Rec_Z", "Gal_z_err", "combo_r_mag", "combo_g_mag", "2MASS21P_k_m",
                    "WISE_w1mpro", "WISE_w1snr", "WISE_w2mpro", "WISE_w2snr",
                    "WISE_w3mpro", "WISE_w3snr"]
    dw_gals = dw_gals.rename(columns=dict(zip(cols_to_rename, cols_renamed)))
    dw_gals["galaxy"] = True

    df_gals = df.merge(dw_gals, on="name", how="outer", suffixes=("", "_dw"))
    for col in dw_gals.columns:
        if col != "name" and col + "_dw" in df_gals.columns:
            df_gals[col] = df_gals[col].combine_first(df_gals[col + "_dw"])
            df_gals.drop(columns=col + "_dw", inplace=True)

    df_gals["galaxy"] = df_gals["galaxy"].fillna(False).astype(bool)

    # Sentinel fix (not in Alec's original): Kim23's raw g/W1/W2/W3 contain
    # unmasked -999 -- apply the project-wide mag<=0 -> NaN rule here so it
    # can never reach a mag_to_flux conversion downstream.
    for mag_col in ["combo_g_mag", "WISE_w1mpro", "WISE_w2mpro", "WISE_w3mpro"]:
        df_gals[mag_col] = df_gals[mag_col].where(df_gals[mag_col] > 0)

    return df_gals
