"""
Problem Type Detector Module
Automatically determines if the problem is classification or regression
"""

import pandas as pd
import numpy as np
from typing import Tuple
from utils.config import CLASSIFICATION_THRESHOLD, REGRESSION_MIN_UNIQUE


class ProblemDetector:
    """
    Detects the type of machine learning problem based on target variable
    """
    
    def __init__(self, target_series: pd.Series):
        """
        Initialize ProblemDetector
        
        Args:
            target_series: Target column as pandas Series
        """
        self.target = target_series
        self.problem_type = None
        self.confidence = None
    
    def detect_problem_type(self) -> Tuple[str, float, dict]:
        """
        Detect whether the problem is classification or regression
        
        Returns:
            Tuple of (problem_type, confidence, details)
            - problem_type: 'classification' or 'regression'
            - confidence: confidence score (0-1)
            - details: dictionary with detection details
        """
        details = self._analyze_target()
        
        # Rule-based detection
        if details['is_numeric']:
            # Numeric target - could be either classification or regression
            if details['n_unique'] <= CLASSIFICATION_THRESHOLD:
                # Few unique values - likely classification
                if details['n_unique'] <= 2:
                    self.problem_type = 'classification'
                    self.confidence = 0.95
                elif details['n_unique'] <= 10:
                    self.problem_type = 'classification'
                    self.confidence = 0.85
                else:
                    self.problem_type = 'classification'
                    self.confidence = 0.70
            else:
                # Many unique values - likely regression
                if details['is_continuous']:
                    self.problem_type = 'regression'
                    self.confidence = 0.90
                else:
                    self.problem_type = 'regression'
                    self.confidence = 0.75
        else:
            # Categorical target - always classification
            self.problem_type = 'classification'
            self.confidence = 0.95
        
        details['detected_type'] = self.problem_type
        details['confidence'] = self.confidence
        
        return self.problem_type, self.confidence, details
    
    def _analyze_target(self) -> dict:
        """
        Analyze target variable characteristics
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'dtype': str(self.target.dtype),
            'n_total': len(self.target),
            'n_unique': self.target.nunique(),
            'n_missing': self.target.isnull().sum(),
            'missing_percentage': (self.target.isnull().sum() / len(self.target) * 100),
            'is_numeric': pd.api.types.is_numeric_dtype(self.target),
            'is_integer': pd.api.types.is_integer_dtype(self.target),
            'is_float': pd.api.types.is_float_dtype(self.target)
        }
        
        # Additional analysis for numeric targets
        if analysis['is_numeric']:
            non_null_target = self.target.dropna()
            
            analysis.update({
                'min': float(non_null_target.min()),
                'max': float(non_null_target.max()),
                'mean': float(non_null_target.mean()),
                'median': float(non_null_target.median()),
                'std': float(non_null_target.std())
            })
            
            # Check if values are continuous or discrete
            # Continuous: has decimal values or wide range relative to unique count
            has_decimals = not all(non_null_target == non_null_target.astype(int))
            range_to_unique_ratio = (analysis['max'] - analysis['min']) / max(analysis['n_unique'], 1)
            
            analysis['is_continuous'] = has_decimals or range_to_unique_ratio > 1
            
            # Get value distribution for small number of unique values
            if analysis['n_unique'] <= 20:
                analysis['value_counts'] = self.target.value_counts().to_dict()
        else:
            # Categorical target
            analysis['value_counts'] = self.target.value_counts().head(20).to_dict()
        
        return analysis
    
    def get_recommendation(self) -> dict:
        """
        Get model recommendations based on problem type
        
        Returns:
            Dictionary with recommendations
        """
        if not self.problem_type:
            self.detect_problem_type()
        
        recommendations = {
            'problem_type': self.problem_type,
            'confidence': self.confidence
        }
        
        if self.problem_type == 'classification':
            recommendations['suggested_models'] = [
                'Logistic Regression',
                'Random Forest',
                'XGBoost',
                'SVM'
            ]
            recommendations['evaluation_metrics'] = [
                'Accuracy',
                'Precision',
                'Recall',
                'F1-Score',
                'ROC-AUC'
            ]
            recommendations['considerations'] = [
                'Check class balance',
                'Consider stratified cross-validation',
                'May need class weighting for imbalanced data'
            ]
        else:  # regression
            recommendations['suggested_models'] = [
                'Linear Regression',
                'Ridge/Lasso',
                'Random Forest Regressor',
                'XGBoost Regressor'
            ]
            recommendations['evaluation_metrics'] = [
                'RMSE',
                'MAE',
                'R² Score'
            ]
            recommendations['considerations'] = [
                'Check for outliers',
                'Consider feature scaling',
                'May benefit from polynomial features'
            ]
        
        return recommendations
    
    def validate_target(self) -> Tuple[bool, str]:
        """
        Validate if target is suitable for ML
        
        Returns:
            Tuple of (is_valid, message)
        """
        if self.target.isnull().all():
            return False, "Target column is entirely missing"
        
        if self.target.isnull().sum() / len(self.target) > 0.5:
            return False, "Target column has more than 50% missing values"
        
        if self.target.nunique() == 1:
            return False, "Target column has only one unique value"
        
        if not pd.api.types.is_numeric_dtype(self.target):
            # Check if categorical target can be encoded
            if self.target.nunique() > 100:
                return False, "Categorical target has too many unique values (>100)"
        
        return True, "Target is valid"
    
    def print_detection_summary(self):
        """Print a formatted summary of the detection"""
        if not self.problem_type:
            self.detect_problem_type()
        
        print("=" * 60)
        print("PROBLEM TYPE DETECTION")
        print("=" * 60)
        
        print(f"\nDetected Type: {self.problem_type.upper()}")
        print(f"Confidence: {self.confidence * 100:.1f}%")
        
        print(f"\nTarget Variable Analysis:")
        print(f"  Unique Values: {self.target.nunique()}")
        print(f"  Data Type: {self.target.dtype}")
        
        if pd.api.types.is_numeric_dtype(self.target):
            print(f"  Range: {self.target.min():.2f} to {self.target.max():.2f}")
        
        rec = self.get_recommendation()
        print(f"\nSuggested Models:")
        for model in rec['suggested_models']:
            print(f"  - {model}")
        
        print("=" * 60)