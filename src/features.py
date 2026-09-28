import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Input: DataFrame
    Output: Dataframe
    creates new features
    """
    df = df.copy()
    
    # Betrag der symmetrischen Features
    df["abs_fAsym"] = df["fAsym"].abs()
    df["abs_fM3Long"] = df["fM3Long"].abs()
    df["abs_fM3Trans"] = df["fM3Trans"].abs()

    # Längen-/Breitenverhältnis
    df["width_length_ratio"] = df["fWidth"] / df["fLength"]

    # Größe der Hillas-Ellipse
    df["ellipse_area"] = df["fLength"] * df["fWidth"] * np.pi

    # wie stark dominiert das hellste Pixel die beiden hellsten Pixel
    df["brightest_pixel_share"] = df["fConc1"] / df["fConc"]

    # Umwandlung von alpha vom Wertebereich 0 bis 90 auf 0 bis 1, 1 entspricht perfekte Ausrichtung zum Kamerazentrum
    df["alpha_alignment"] = np.cos(np.deg2rad(df["fAlpha"]))

    return(df)