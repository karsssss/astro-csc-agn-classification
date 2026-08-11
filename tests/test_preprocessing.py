import os

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from src.preprocessing import clean_types, mask_magnitude_sentinels, merge_kim23_galaxies


def test_clean_types_coerces_bools_and_blank_strings():
    df = pd.DataFrame({
        "conf_flag": [True, np.nan, False],
        "extent_flag": [np.nan, True, False],
        "xray_agn": [True, False, np.nan],
        "optical_agn": [False, np.nan, True],
        "ir_agn": [np.nan, False, True],
        "some_text": ["  hello  ", "", "world"],
    })

    cleaned = clean_types(df)

    assert cleaned["conf_flag"].tolist() == [True, False, False]
    assert cleaned["extent_flag"].tolist() == [False, True, False]
    assert cleaned["xray_agn"].dtype == bool
    assert cleaned["optical_agn"].tolist() == [False, False, True]
    assert cleaned["ir_agn"].tolist() == [False, False, True]
    assert cleaned["some_text"].iloc[0] == "hello"
    assert pd.isna(cleaned["some_text"].iloc[1])
    assert cleaned["some_text"].iloc[2] == "world"


def test_mask_magnitude_sentinels_preserves_positive_converts_nonpositive():
    df = pd.DataFrame({
        "PS21P_g": [18.5, -999.0, 0.0, 20.1],
        "LEG21P_G": [-9.0, 17.2, 19.0, -999.0],
    })
    zero_point_jy = {"PS21P_g": 3631.0, "LEG21P_G": 3631.0}

    masked = mask_magnitude_sentinels(df, zero_point_jy)

    assert masked["PS21P_g"].tolist()[0] == 18.5
    assert masked["PS21P_g"].tolist()[3] == 20.1
    assert masked["PS21P_g"].isna().tolist()[1:3] == [True, True]

    assert masked["LEG21P_G"].tolist()[1] == 17.2
    assert masked["LEG21P_G"].tolist()[2] == 19.0
    assert pd.isna(masked["LEG21P_G"].iloc[0])
    assert pd.isna(masked["LEG21P_G"].iloc[3])


def test_merge_kim23_galaxies_outer_join_and_combine_first(tmp_path):
    kim_table = Table({
        "CSCID": ["2CXO_J000001.0+000001", "2CXO_J000002.0+000002"],
        "SDSSID": ["1", "2"], "PSID": ["1", "2"], "LEGSID": ["1", "2"],
        "WISEID": ["1", "2"], "2MASSID": ["1", "2"], "GALEXID": ["1", "2"],
        "z_code": ["spec", "phot"], "class_SDSS": ["GALAXY", "GALAXY"],
        "subclass_SDSS": ["", ""],
        "ra": [0.0, 0.5], "dec": [0.0, 0.5],
        "z": [0.05, 0.10], "zErr": [0.001, 0.002],
        "r": [18.0, 19.0], "g": [18.5, -999.0], "k": [14.0, 14.5],
        "W1": [13.0, -999.0], "w1snr": [10.0, 5.0],
        "W2": [12.5, 11.0], "w2snr": [9.0, 8.0],
        "W3": [9.0, 8.5], "w3snr": [3.0, 2.0],
        "significance": [5.0, 6.0],
        "FxB15": [1.0, 2.0], "FxH15": [0.5, 1.0],
        "spec": np.array([1, 0], dtype=np.int16),
    })
    kim_path = tmp_path / "kim23_test.fits"
    kim_table.write(kim_path, format="fits")

    df = pd.DataFrame({
        "name": ["2CXO_J000001.0+000001", "2CXO_J999999.0+999999"],
        "ra": [0.0, 10.0], "dec": [0.0, 10.0],
        "significance": [99.0, np.nan],  # row 0 already populated -- must survive
        "WISE_w1mpro": [np.nan, 15.0],   # row 0 missing -- must be backfilled
    })

    merged = merge_kim23_galaxies(df, str(kim_path))

    # outer join: the 2nd Kim23 row (...000002...) isn't in df -> 1 new row
    assert len(merged) == 3

    row0 = merged[merged["name"] == "2CXO_J000001.0+000001"].iloc[0]
    # combine_first keeps df's existing non-null value, does not overwrite it
    assert row0["significance"] == 99.0
    # combine_first fills df's NaN from Kim23's (renamed) WISE_w1mpro
    assert row0["WISE_w1mpro"] == 13.0

    # galaxy is a proper bool: True only for Kim23-matched/added rows
    assert merged.set_index("name")["galaxy"].to_dict() == {
        "2CXO_J000001.0+000001": True,
        "2CXO_J999999.0+999999": False,
        "2CXO_J000002.0+000002": True,
    }

    # row1 (...000002...) came in only via Kim23's -999 sentinel for W1 ->
    # must be masked to NaN, not left as -999
    row1 = merged[merged["name"] == "2CXO_J000002.0+000002"].iloc[0]
    assert pd.isna(row1["WISE_w1mpro"])


def test_merge_kim23_galaxies_real_file_matched_rows_backfill_no_new_rows():
    """Regression check against the actual data file: for CSCIDs known to
    already exist in the catalog, the outer join backfills in place with
    0 row-count change (all 8,547 Kim23_CGC.fits rows currently match an
    existing catalog `name` -- see merge_kim23_galaxies's docstring)."""
    kim_path = "data/raw/Kim23_CGC.fits"
    if not os.path.exists(kim_path):
        pytest.skip("data/raw/Kim23_CGC.fits not present")

    real_names = Table.read(kim_path)["CSCID"]  # all 8,547 -- must cover every Kim23 name
    real_names = [
        (n.decode() if isinstance(n, bytes) else n).strip() for n in real_names
    ]

    df = pd.DataFrame({"name": real_names, "significance": [1.0] * len(real_names)})
    merged = merge_kim23_galaxies(df, kim_path)

    assert len(merged) == len(df)  # matched rows, not new -- no row-count change
    assert merged["galaxy"].all()
    # combine_first kept df's existing significance rather than overwriting it
    assert (merged["significance"] == 1.0).all()
