import pandas as pd
import numpy as np

def wrangle(filepath):
    """Loads and preprocess the Seoul real estate

    Args:
        filepath (str): Path to the CSV file

    Returns:
    pd.Dataframe
        Cleaned dataframe ready for modeling
    """
    df = pd.read_csv(filepath)

    # Removing rows with missing sales values (target variable)
    df = df.dropna(subset = ["min_sales", "max_sales", "avg_sales"])

    # Removing rows with impossible area or floors
    df = df[(df["m2"] > 0) & (df["p"] > 0)]

    # Extracting year and month into new columns
    df["build_year"] = df["buildDate"] // 100
    df["build_month"] = df["buildDate"] % 100

    # Validating: month is value from 1-12, and year is realistic
    df = df[(df['build_month'] >= 1) & (df['build_month'] <= 12)]
    df = df[(df['build_year'] >= 1900) & (df['build_year'] <= 2026)]

    current_year = 2026
    df["building_age"] = current_year - df["build_year"]

    # Removing negative ages (future buildings) if any
    df = df[df["building_age"] >= 0]

    # dropping min_sales and max_sales to prevent leakage
    # dropping id to prevent over-fitting
    df = df.drop(columns = ["id", "min_sales", "max_sales", "buildDate"])

    # creating new column 'dist_to_gangnam'
    gangnam_lat, gangnam_lng = 37.4979, 127.0276
    def haversine_distance(lat1, lng1, lat2, lng2):
        earth_radius_km = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lng2 - lng1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    df['dist_to_gangnam'] = haversine_distance(df['lat'], df['lng'], gangnam_lat, gangnam_lng)

    # resetting index
    df = df.reset_index(drop = True)
    return df