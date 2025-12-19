"""
Preprocessor Module
Automated data preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List
from utils.config import MISSING_THRESHOLD, VARIANCE_THRESHOLD


class AutoPreprocessor:
    """
    Automated preprocessing pipeline that handles missing values,
    encoding, scaling, and feature selection
    """
    
    def __init__(self, problem_type: str):
        """
        Initialize AutoPreprocessor
        
        Args:
            problem_type: 'classification' or 'regression'
        """
        self.problem_type = problem_type
        self.numerical_features = []
        self.categorical_features = []
        self.features_to_drop = []
        self.target_encoder = None
        self.preprocessing_pipeline = None
        self.feature_names = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoPreprocessor':
        """
        Fit the preprocessing pipeline
        
        Args:
            X: Feature dataframe
            y: Target series
        
        Returns:
            Self
        """
        # Identify feature types
        self._identify_feature_types(X)
        
        # Identify features to drop
        self._identify_features_to_drop(X)
        
        # Remove features to drop
        X_cleaned = X.drop(columns=self.features_to_drop)
        self._identify_feature_types(X_cleaned)
        
        # Build preprocessing pipeline
        self._build_pipeline()
        
        # Fit the pipeline
        if len(self.numerical_features) > 0 or len(self.categorical_features) > 0:
            self.preprocessing_pipeline.fit(X_cleaned)
            
            # Store feature names after transformation
            self.feature_names = self._get_feature_names(X_cleaned)
        
        # Handle target encoding for classification
        if self.problem_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(y)
        
        return self
    
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features and target
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
        
        Returns:
            Tuple of (transformed_X, transformed_y)
        """
        # Remove features to drop
        X_cleaned = X.drop(columns=self.features_to_drop, errors='ignore')
        
        # Transform features
        if self.preprocessing_pipeline:
            X_transformed = self.preprocessing_pipeline.transform(X_cleaned)
        else:
            X_transformed = X_cleaned.values
        
        # Transform target if provided
        y_transformed = None
        if y is not None:
            if self.target_encoder:
                y_transformed = self.target_encoder.transform(y)
            else:
                y_transformed = y.values
        
        return X_transformed, y_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step
        
        Args:
            X: Feature dataframe
            y: Target series
        
        Returns:
            Tuple of (transformed_X, transformed_y)
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def _identify_feature_types(self, X: pd.DataFrame):
        """Identify numerical and categorical features"""
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def _identify_features_to_drop(self, X: pd.DataFrame):
        """
        Identify features that should be dropped:
        - Too many missing values
        - Zero variance
        - Too many unique values for categorical
        """
        self.features_to_drop = []
        
        # Drop columns with too many missing values
        missing_pct = X.isnull().sum() / len(X)
        high_missing_cols = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
        self.features_to_drop.extend(high_missing_cols)
        
        # Drop zero variance numerical columns
        for col in self.numerical_features:
            if X[col].nunique() == 1:
                self.features_to_drop.append(col)
        
        # Drop categorical columns with too many unique values
        for col in self.categorical_features:
            if X[col].nunique() > 50:  # Arbitrary threshold
                self.features_to_drop.append(col)
        
        self.features_to_drop = list(set(self.features_to_drop))
    
    def _build_pipeline(self):
        """Build the preprocessing pipeline"""
        transformers = []
        
        # Numerical pipeline
        if len(self.numerical_features) > 0:
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_pipeline, self.numerical_features))
        
        # Categorical pipeline
        if len(self.categorical_features) > 0:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoderWithNames())
            ])
            transformers.append(('cat', categorical_pipeline, self.categorical_features))
        
        if transformers:
            self.preprocessing_pipeline = ColumnTransformer(
                transformers=transformers,
                remainder='drop'
            )
    
    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get feature names after transformation"""
        feature_names = []
        
        # Numerical features keep their names
        feature_names.extend(self.numerical_features)
        
        # Categorical features are one-hot encoded
        if len(self.categorical_features) > 0:
            for col in self.categorical_features:
                unique_values = X[col].unique()
                for val in unique_values:
                    if pd.notna(val):
                        feature_names.append(f"{col}_{val}")
        
        return feature_names
    
    def get_preprocessing_info(self) -> dict:
        """Get information about the preprocessing steps"""
        info = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'features_dropped': self.features_to_drop,
            'n_features_in': len(self.numerical_features) + len(self.categorical_features),
            'n_features_out': len(self.feature_names),
            'target_encoded': self.target_encoder is not None
        }
        
        if self.target_encoder:
            info['target_classes'] = self.target_encoder.classes_.tolist()
        
        return info


class OneHotEncoderWithNames:
    """
    Simple one-hot encoder that maintains compatibility with sklearn pipeline
    """
    
    def __init__(self):
        self.categories_ = {}
    
    def fit(self, X, y=None):
        """Fit the encoder"""
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                self.categories_[col] = sorted(X[col].dropna().unique())
        else:
            # Handle numpy array
            for i in range(X.shape[1]):
                self.categories_[i] = sorted(pd.Series(X[:, i]).dropna().unique())
        return self
    
    def transform(self, X):
        """Transform the data"""
        if isinstance(X, pd.DataFrame):
            return pd.get_dummies(X, columns=X.columns).values
        else:
            # Handle numpy array
            df = pd.DataFrame(X)
            return pd.get_dummies(df, columns=df.columns).values
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        return self.fit(X, y).transform(X)