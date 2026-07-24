import numpy as np
import pandas as pd

from src.preprocessing import clean_types, mask_magnitude_sentinels


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
