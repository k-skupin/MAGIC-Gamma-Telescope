import numpy as np

def engineer_features(df):
    """
    Input: DataFrame
    
    Output: Dataframe
    creates new features
    """
    df = df.copy()

    
    # *Absolute values of the symmetric features*
    df["abs_fAsym"] = df["fAsym"].abs()
    df["abs_fM3Long"] = df["fM3Long"].abs()
    df["abs_fM3Trans"] = df["fM3Trans"].abs()

    # length/width ratio
    df["width_length_ratio"] = df["fWidth"] / df["fLength"]

    # size of Hillas-elypse
    df["ellipse_area"] = df["fLength"] * df["fWidth"] * np.pi

    # How strongly the brightest pixel dominates the two brightest pixels
    df["brightest_pixel_share"] = df["fConc1"] / df["fConc"]

    # Transformation of alpha from the range 0 to 90 into the range 0 to 1, where 1 represents perfect alignment with the camera center.
    df["alpha_alignment"] = np.cos(np.deg2rad(df["fAlpha"]))

    return(df)