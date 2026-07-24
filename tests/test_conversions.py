from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import preprocessing
from src.conversions import (
    ZERO_POINT_JY,
    build_feature_matrix,
    combine_first_column,
    cross_check_mag_zero_point,
    mag_to_flux_jy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CATALOG_PATH = REPO_ROOT / "data" / "raw" / "Multiwave_UMAP_6_26_26.csv"


def test_mag_to_flux_jy_round_trip():
    zp = 3631.0
    mag = pd.Series([15.0, 18.5, 21.2, 23.0])

    flux = mag_to_flux_jy(mag, zp)
    mag_recovered = -2.5 * np.log10(flux / zp)

    assert np.allclose(mag_recovered, mag, atol=1e-10)


def test_mag_to_flux_jy_masks_sentinels():
    mag = pd.Series([-999.0, -9.0, 0.0, 18.0])
    flux = mag_to_flux_jy(mag, 3631.0)

    assert flux.iloc[:3].isna().all()
    assert flux.iloc[3] > 0


def test_cross_check_mag_zero_point_reports_small_offset_for_matched_bands():
    # ref_mag = 22.5 - 2.5*log10(nanomaggie) -- construct nanomaggie fluxes
    # that correspond exactly to the given PS21P_r magnitudes, so the true
    # offset is 0 and any median offset above a small tolerance would flag a
    # real zero-point/unit mismatch.
    r_mags = pd.Series([16.0, 17.5, 19.0, 20.5, 22.0])
    nanomaggie = 10 ** ((22.5 - r_mags) / 2.5)
    df = pd.DataFrame({"PS21P_r": r_mags, "SDSS_CALIBFLUX_r": nanomaggie})

    diff = cross_check_mag_zero_point(df, "PS21P_r", "SDSS_CALIBFLUX_r", label="test fixture")

    assert len(diff) == 5
    assert abs(diff.median()) < 0.05


def test_build_feature_matrix_has_no_raw_flux_columns():
    zero_points = {"PS21P_g": 3631.0, "PS21P_r": 3631.0, "WISE_w1mpro": 309.540}
    n = 200
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "significance": rng.uniform(3, 50, n),
        "flux_aper_b": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_lolim_b": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_hilim_b": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_h": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_lolim_h": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_hilim_h": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_s": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_lolim_s": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_hilim_s": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_m": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_lolim_m": rng.uniform(1e-15, 1e-13, n),
        "flux_aper_hilim_m": rng.uniform(1e-15, 1e-13, n),
        "combo_g": rng.uniform(16, 22, n),
        "combo_r": rng.uniform(16, 22, n),
        "combo_z_mag": rng.uniform(16, 22, n),
        "WISE_w1mpro": rng.uniform(12, 17, n),
        "WISE_w1snr": rng.uniform(3, 50, n),
        "WISE_w2mpro": rng.uniform(12, 16, n),
        "WISE_w2snr": rng.uniform(3, 50, n),
        "WISE_w3mpro": rng.uniform(8, 12, n),
        "WISE_w3snr": rng.uniform(3, 50, n),
        "hard_hs": rng.uniform(-1, 1, n),
        "hard_hs_lolim": rng.uniform(-1, 1, n),
        "hard_hs_hilim": rng.uniform(-1, 1, n),
        "hard_hm": rng.uniform(-1, 1, n),
        "hard_hm_lolim": rng.uniform(-1, 1, n),
        "hard_hm_hilim": rng.uniform(-1, 1, n),
        "hard_ms": rng.uniform(-1, 1, n),
        "hard_ms_lolim": rng.uniform(-1, 1, n),
        "hard_ms_hilim": rng.uniform(-1, 1, n),
    })

    _, X = build_feature_matrix(df)

    leak_hits = [
        c for c in X.columns
        if c.startswith(("flux_aper", "flux_combo", "flux_wise", "flux_WISE"))
    ]
    assert not leak_hits, f"raw flux columns leaked into feature matrix: {leak_hits}"


@pytest.mark.skipif(not RAW_CATALOG_PATH.exists(), reason="raw catalog not available")
def test_build_feature_matrix_row_count_matches_v_final():
    df = preprocessing.load_catalog(RAW_CATALOG_PATH)
    df = preprocessing.mask_magnitude_sentinels(df, ZERO_POINT_JY)
    df = combine_first_column(df, "PS21P_g", "LEG21P_G", "combo_g")
    df = combine_first_column(df, "PS21P_r", "LEG21P_R", "combo_r")
    df = combine_first_column(df, "PS21P_z", "LEG21P_Z", "combo_z_mag")

    _, X = build_feature_matrix(df)

    assert abs(len(X) - 25801) <= 5
    assert X.shape[1] == 22
