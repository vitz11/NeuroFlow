"""
Data Profiler Module
Analyzes dataset characteristics and quality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from utils.helpers import detect_column_types, calculate_missing_percentage, get_basic_stats


class DataProfiler:
    """
    Profiles datasets to understand their characteristics
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataProfiler
        
        Args:
            df: Input dataframe to profile
        """
        self.df = df
        self.profile = {}
    
    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate comprehensive data profile
        
        Returns:
            Dictionary containing profile information
        """
        self.profile = {
            'basic_stats': self._get_basic_stats(),
            'column_types': self._get_column_types(),
            'missing_values': self._analyze_missing_values(),
            'duplicates': self._analyze_duplicates(),
            'numerical_stats': self._get_numerical_stats(),
            'categorical_stats': self._get_categorical_stats(),
            'data_quality_score': self._calculate_quality_score()
        }
        
        return self.profile
    
    def _get_basic_stats(self) -> Dict[str, Any]:
        """Get basic dataset statistics"""
        return get_basic_stats(self.df)
    
    def _get_column_types(self) -> Dict[str, List[str]]:
        """Detect column types"""
        return detect_column_types(self.df)
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset
        
        Returns:
            Dictionary with missing value analysis
        """
        missing_pct = calculate_missing_percentage(self.df)
        
        return {
            'total_missing': self.df.isnull().sum().sum(),
            'columns_with_missing': missing_pct[missing_pct > 0].to_dict(),
            'percentage_by_column': missing_pct.to_dict(),
            'columns_mostly_missing': missing_pct[missing_pct > 50].index.tolist()
        }
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """
        Analyze duplicate rows
        
        Returns:
            Dictionary with duplicate analysis
        """
        n_duplicates = self.df.duplicated().sum()
        
        return {
            'n_duplicates': int(n_duplicates),
            'duplicate_percentage': float(n_duplicates / len(self.df) * 100) if len(self.df) > 0 else 0
        }
    
    def _get_numerical_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for numerical columns
        
        Returns:
            Dictionary with statistics for each numerical column
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numerical_cols:
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'n_unique': int(self.df[col].nunique()),
                'skewness': float(self.df[col].skew()),
                'kurtosis': float(self.df[col].kurtosis())
            }
        
        return stats
    
    def _get_categorical_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for categorical columns
        
        Returns:
            Dictionary with statistics for each categorical column
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        stats = {}
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            stats[col] = {
                'n_unique': int(self.df[col].nunique()),
                'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_common_freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'top_5_values': value_counts.head(5).to_dict()
            }
        
        return stats
    
    def _calculate_quality_score(self) -> float:
        """
        Calculate overall data quality score (0-100)
        
        Returns:
            Quality score
        """
        score = 100.0
        
        # Deduct for missing values
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100)
        score -= missing_pct * 0.5
        
        # Deduct for duplicates
        duplicate_pct = (self.df.duplicated().sum() / len(self.df) * 100) if len(self.df) > 0 else 0
        score -= duplicate_pct * 0.3
        
        # Deduct if dataset is too small
        if len(self.df) < 100:
            score -= 10
        elif len(self.df) < 500:
            score -= 5
        
        return max(0, min(100, score))
    
    def get_target_column_info(self, target_col: str) -> Dict[str, Any]:
        """
        Get detailed information about the target column
        
        Args:
            target_col: Name of the target column
        
        Returns:
            Dictionary with target column information
        """
        if target_col not in self.df.columns:
            return {'error': 'Target column not found'}
        
        target_series = self.df[target_col]
        
        info = {
            'dtype': str(target_series.dtype),
            'n_unique': int(target_series.nunique()),
            'missing_count': int(target_series.isnull().sum()),
            'missing_percentage': float(target_series.isnull().sum() / len(target_series) * 100)
        }
        
        # Add type-specific information
        if pd.api.types.is_numeric_dtype(target_series):
            info.update({
                'mean': float(target_series.mean()),
                'median': float(target_series.median()),
                'min': float(target_series.min()),
                'max': float(target_series.max()),
                'std': float(target_series.std())
            })
        else:
            value_counts = target_series.value_counts()
            info.update({
                'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'value_counts': value_counts.head(10).to_dict()
            })
        
        return info
    
    def print_summary(self):
        """Print a formatted summary of the profile"""
        if not self.profile:
            self.generate_profile()
        
        print("=" * 60)
        print("DATA PROFILE SUMMARY")
        print("=" * 60)
        
        basic = self.profile['basic_stats']
        print(f"\nBasic Statistics:")
        print(f"  Rows: {basic['n_rows']:,}")
        print(f"  Columns: {basic['n_columns']}")
        print(f"  Duplicates: {basic['n_duplicates']:,}")
        print(f"  Missing Cells: {basic['missing_cells']:,} ({basic['missing_percentage']:.2f}%)")
        print(f"  Memory Usage: {basic['memory_usage_mb']:.2f} MB")
        
        col_types = self.profile['column_types']
        print(f"\nColumn Types:")
        print(f"  Numerical: {len(col_types['numerical'])}")
        print(f"  Categorical: {len(col_types['categorical'])}")
        
        print(f"\nData Quality Score: {self.profile['data_quality_score']:.2f}/100")
        
        print("=" * 60)