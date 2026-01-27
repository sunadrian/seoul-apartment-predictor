"""
Tests for the data_processing.py module
Tests the wrangle() function for Seoul apartment data cleaning
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys
import os
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to project root, then into src
src_path = os.path.join(current_dir, '..', 'src')
# Convert to absolute path and add to Python path
sys.path.insert(0, os.path.abspath(src_path))

# current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
# src_path = os.path.join(current_dir, '..', 'src')
# sys.path.insert(0, os.path.abspath(src_path))


from data_processing import wrangle

class TestWrangleFunction:
    """Test suite for the wrangle function"""

    @pytest.fixture
    def sample_data(self):
        """Creates sample data

        Returns:
            Dataframe: sample data
        """
        data = {
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'lat': [37.580, 37.6, 37.4, 37.7, 37, 37.6, 37.4, 37.731, 37.5, 37.6],
            'lng': [127.0, 126.9, 127.1, 126.8, 127.0919, 126.900, 127.001, 126.8, 127.0, 126.9],
            'households': [900, 200, 4000, 300, 100, 200, 10900, 1, 100, 200],
            'buildDate': [201501, 200812, 199506, 202201, 201501, 200812, 199506, 202201, 201501, 200812],
            'score': [4.5, 3.8, 4.2, 4.8, 5, 3.8, 4.2, 0, 4.5, 3.8],
            'm2': [84.5, 1, 0, 73.8, 84.5, 59.2, 102.3, 73.8, 84.5, 300.2],
            'p': [15, 8, 22, 5, 15, 99, 22, 5, 1, 8],
            'min_sales': [300000000, 250000000, 400000000, 280000000, 300000000, 250000000, 400000000, 280000000, 300000000, 250000000],
            'max_sales': [500000000, 350000000, 600000000, 320000000, 500000000, 350000000, 600000000, 320000000, 500000000, 350000000],
            'avg_sales': [400000000, 300000000, 500000000, 300000000, 400000000, 300000000, 500000000, 300000000, 400000000, 300000000]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def edge_case_data(self):
        """Create data with various edge cases

        Returns:
            Dataframe: sample edge case data
        """
        data = {
            'id': [1, 2, 3, 4, 5, 6, 7, 8],
            'lat': [37.5, 37.6, 37.4, 37.7, 37.5, 37.6, 37.4, 37.7],
            'lng': [127.0, 126.9, 127.1, 126.8, 127.0, 126.9, 127.1, 126.8],
            'households': [100, 200, 150, 300, 100, 200, 150, 300],
            'buildDate': [201501, 200800, 199513, 189901, 202701, 203001, 201501, 201501],
            'score': [4.5, 3.8, 4.2, 4.8, 4.5, 3.8, 4.2, 4.8],
            'm2': [84.5, -10, 0, 73.8, 84.5, 59.2, 102.3, 1000],
            'p': [15, 8, -5, 0, 15, 8, 22, 50],
            'min_sales': [300, np.nan, 400, 280, 300, 250, 400, 280],
            'max_sales': [500, 350, np.nan, 320, 500, 350, 600, 320],
            'avg_sales': [400, 300, np.nan, 300, 400, 300, 500, 1000000]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_csv(self, sample_data, tmp_path):
        """Create a temporary CSV file for testing

        Args:
            data (pandas Dataframe): sample data
            path (str): temporary path

        Returns:
            str: csv filepath
        """
        filepath = tmp_path/"test_data.csv"
        sample_data.to_csv(filepath, index = False)
        return str(filepath)
    

    def test_functionality(self, temp_csv):
        """Test basic functionality of wrangle()

        Args:
            filepath (str): csv filepath
        """
        result = wrangle(temp_csv)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        assert result.index[0] == 0
        assert result.index.is_monotonic_increasing

        assert not result.duplicated().any()
        print("functionality ok")

    def test_column_operations(self, temp_csv):
        """Test that correct columns are preserved, added, and removed

        Args:
            filepath (str): csv filepath
        """
        result = wrangle(temp_csv)

        # columns that should be removed
        removed_cols = ['id', 'min_sales', 'max_sales', 'buildDate']
        for col in removed_cols:
            assert col not in result.columns, f"Column '{col}' should be removed"

        # columns that should be preserved
        preserved_cols = ['lat', 'lng', 'households', 'score', 'm2', 'p', 'avg_sales']
        for col in preserved_cols:
            assert col in result.columns, f"Column '{col}' should be preserved"

        # columns that should be added
        added_cols = ['build_year', 'build_month', 'building_age']
        for col in added_cols:
            assert col in result.columns, f"Column '{col}' should be added"
        print("col ops ok")

    def test_feat_engineering_calc(self, temp_csv):
        """Verify all calculations and transformations

        Args:
            filepath (str): csv filepath
        """
        result = wrangle(temp_csv)

        # Test build_year and build_month extraction for 
        # buildDate = 201501: year = 2015, month = 1
        assert result.loc[0, 'build_year'] == 2015
        assert result.loc[0, 'build_month'] == 1

        # buildDate = 200812: year = 2008, month = 12
        assert result.loc[1, 'build_year'] == 2008
        assert result.loc[1, 'build_month'] == 12

        # Test building_age calculation
        current_year = 2026
        assert result.loc[0, 'building_age'] == current_year - 2015  # 11 years
        assert result.loc[1, 'building_age'] == current_year - 2008  # 18 years

        # Verify datatypes of engineered columns
        assert pd.api.types.is_integer_dtype(result['build_year'])
        assert pd.api.types.is_integer_dtype(result['build_month'])
        assert pd.api.types.is_integer_dtype(result['building_age'])

        print("feat engineer ok")

    def test_data_validation_and_filtering(self, edge_case_data):
        """Test all data validation rules and filtering logic.

        Args:
            edge_case_data (Dataframe): edge case pandas Dataframe
        """
        # Create temp file with edge case data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            edge_case_data.to_csv(f.name, index=False)
            filepath = f.name
        
        try:
            result = wrangle(filepath)
            
            # Should only keep valid rows (row 0, maybe 6 depending on sales)
            assert len(result) >= 1
            
            # No invalid areas or floors
            assert (result['m2'] > 0).all()
            assert (result['p'] > 0).all()
            
            # No missing sales values
            assert result[['avg_sales']].notnull().all().all()
            
            # Valid build dates
            assert (result['build_month'] >= 1).all()
            assert (result['build_month'] <= 12).all()
            assert (result['build_year'] >= 1900).all()
            assert (result['build_year'] <= 2026).all()
            
            # No negative building ages
            assert (result['building_age'] >= 0).all()
            
        finally:
            os.unlink(filepath)
        
        print("validate ok")