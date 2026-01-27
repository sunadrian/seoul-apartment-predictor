import pandas as pd
import numpy as np

def wrangle(filepath):
    """_summary_
    Loads and preprocess the Seoul real estate

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
    
if __name__ == "__main__":
    test_data = pd.DataFrame({
        'id': [1, 2],
        'lat': [37.5, 37.6],
        'lng': [127.0, 126.9],
        'households': [100, 200],
        'buildDate': [201501, 200812],
        'score': [4.5, 3.8],
        'm2': [84.5, 59.2],
        'p': [15, 8],
        'min_sales': [300000000, 250000000],
        'max_sales': [500000000, 350000000],
        'avg_sales': [400000000, 300000000]
    })
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        filepath = f.name
    
    try:
        result = wrangle(filepath)
        print(f"   Result shape: {result.shape}")
        print(f"   Columns: {list(result.columns)}")
        print(f"   Sample building_age: {result.iloc[0]['building_age']}")
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")