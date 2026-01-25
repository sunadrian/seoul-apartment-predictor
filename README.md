# Seoul Apartment Price Predictor

## Overview: 
Predict apartment prices in Seoul using machine learning models trained on Seoul Apartment Price Data from Kaggle. This project analyzes various features affecting apartment prices and builds predictive models for accurate price estimation.

## Dataset:
- **Source**: [Kaggle - Seoul Real Estate Datasets] https://www.kaggle.com/datasets/jcy1996/seoul-real-estate-datasets
- **Size**: 4,021 rows, 11 columns
- **Key Features**: Coordinates (lat/lng), area (m^2), floor count (p), households, , building date, quality score
- **Target**: `avg_sales` - Average apartment price in complex (Korean Won)

## Data Preprocessing

**Row filtering**:
- Removed apartments with area ≤ 0 m^2 ('m2' column)
- Removed apartments with floor count ≤ 0 ('p' column)
- Removed rows missing target value

## Feature Development

###  A. Feature Selection
** From 11 original columns, I kept 7:**
- Location: `lat`, `lng`
- Physical: `m2`, `p`, `households`
- Quality: `score`

**Removed 3 columns:**
- `id`: Non-predictive identifier
- `min_sales`, `max_sales`: Prevent data leakage
- `build_date`: Redundant with build_year and build_month

### B. Feature Engineering
**Added 3 engineered features:**
- 'build_year': the year the apartment was built
- 'build_month': the month the apartment was built
- 'building_age': the age of the building (as of 2026)

### C. Final Feature Set
**10 total features for modeling**
- 6 original features
- 3 engineered features
- 1 target (`avg_sales`)

## Feature Details
- lat,lng: latitude and longitude of the apartment respectfully
- m²: area in square meters
- p: number of floors
- households: number of households in residence
- score: total evaluation (out of 5 stars)