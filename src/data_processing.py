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

    # resetting index
    df = df.reset_index(drop = True)
    return df